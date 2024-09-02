""" BaseServer for FedETuning """

import os
import random
import threading
from abc import ABC

import torch

import wandb
from fedlab.core.coordinator import Coordinator
from fedlab.core.server.handler import (Aggregators,
                                        ParameterServerBackendHandler)
from fedlab.core.server.manager import ServerManager
from fedlab.utils import MessageCode
from fedlab.utils.serialization import SerializationTool
from utils.register import registry


class BaseSyncServerHandler(ParameterServerBackendHandler, ABC):
    def __init__(self, model, valid_data, test_data):

        self.valid_data = valid_data
        self.test_data = test_data

        config = registry.get("config")
        self.model_config = config.model_config
        self.data_config = config.data_config
        self.training_config = config.training_config
        self.federated_config = config.federated_config

        self.logger = registry.get("logger")

        self.device = config.training_config.device
        self._model = model.to(self.device)
        self._group_num = 10
        
        self._group_models = [model.to(self.device) for _ in range(self._group_num)]

        # basic setting
        self.client_num_in_total = config.federated_config.clients_num
        self.sample_ratio = config.federated_config.sample

        self._group_size = self.client_num_in_total / self._group_num

        # client buffer
        self.client_buffer_cache = []
        self.cache_cnt = 0

        # stop condition
        self.global_round = config.federated_config.rounds
        self.round = 0

        #  metrics & eval
        self._build_metric()
        self._build_eval()
        self._setup_wandb()
        self.global_valid_best_metric = (
            float("inf")
            if self.training_config.is_decreased_valid_metric
            else -float("inf")
        )
        self.global_test_best_metric = 0.0
        self.metric_log = {
            "model_type": self.model_config.model_type,
            "clients_num": self.federated_config.clients_num,
            "alpha": self.federated_config.alpha,
            "task": self.data_config.task_name,
            "fl_algorithm": self.federated_config.fl_algorithm,
            "info": f"{self.model_config.model_type}_{self.federated_config.fl_algorithm}_"
            f"{self.federated_config.clients_num}_{self.federated_config.alpha}",
            "logs": [],
        }
        # metric line
        self.metric_name = self.metric.metric_name
        times = registry.get("run_time")
        if self.training_config.do_grid:
            grid_info = registry.get("grid_info")
            self.metric_line = (
                f"{times}_{self.model_config.model_type}_{self.training_config.tuning_type}_"
                f"seed={self.training_config.seed}_rounds={self.federated_config.rounds}_"
                f"cli={self.federated_config.clients_num}_alp={self.federated_config.alpha}_"
                f"sap={self.federated_config.sample}_epo={self.training_config.num_train_epochs}_"
                f"lr={self.training_config.learning_rate}_{grid_info}_"
            )
        else:
            self.metric_line = (
                f"{times}_{self.model_config.model_type}_{self.training_config.tuning_type}_"
                f"seed={self.training_config.seed}_rounds={self.federated_config.rounds}_"
                f"cli={self.federated_config.clients_num}_alp={self.federated_config.alpha}_"
                f"sap={self.federated_config.sample}_lr={self.training_config.learning_rate}_"
                f"epo={self.training_config.num_train_epochs}_"
            )
        # global model
        self.glo_save_file = os.path.join(
            self.training_config.checkpoint_dir,
            f"{times}_{self.model_config.model_type}.pth",
        )
        self.best_glo_params = None

    def _build_eval(self):
        self.eval = registry.get_eval_class(self.training_config.metric_name)(
            self.device, self.metric
        )

    def _build_metric(self):
        self.metric = registry.get_metric_class(self.training_config.metric_name)(
            self.data_config.task_name, self.training_config.is_decreased_valid_metric
        )

    def _setup_wandb(self):
        from datetime import datetime

        # 获取当前时间
        now = datetime.now()

        # 将时间格式化为字符串
        time_str = now.strftime("%Y-%m-%d %H:%M")
        name = (
            f"client={self.federated_config.clients_num}_alpha={self.federated_config.alpha}_sap={self.federated_config.sample}_epoch={self.training_config.num_train_epochs}_"
            + time_str
        )
        # project_name = f"{self.data_config.task_name}-{self.model_config.model_type}"
        # project_name = "client-num-test-qnli-roberta"
        # project_name = "sfl-bert-qnli"
        # project_name = "legend-qnli"
        # project_name = "legend-pre-sst-2-position"
        project_name = "non-iid-fedlora-local-training-interval"
        wandb.init(project=project_name, name=name)

    def stop_condition(self) -> bool:
        return self.round >= self.global_round

    def sample_clients(self):
        selection = random.sample(
            range(self.client_num_in_total), self.client_num_per_round
        )
        return selection

    def grouped_clients(self, num_group):
        group_sizes = [int(self.client_num_in_total / num_group) for _ in range(num_group)] # [5,5]
        all_ids = [id for id in range(self.client_num_in_total)]
        group_ids = [all_ids[i: i + group_sizes[0]] for i in range(0, len(all_ids), group_sizes[0])] # [[0,1,2,3,4], [5,6,7,8,9]]
        return group_ids
    
    def _update_global_model(self, payload):
        assert len(payload) > 0

        if len(payload) == 1:
            self.client_buffer_cache.append(payload[0].clone())
        else:
            self.client_buffer_cache += payload  # serial trainer

        assert len(self.client_buffer_cache) <= self.client_num_per_round

        if len(self.client_buffer_cache) == self.client_num_per_round:
            model_parameters_list = self.client_buffer_cache
            self.logger.debug(
                f"Model parameters aggregation, number of aggregation elements {len(model_parameters_list)}"
            )

            # use aggregator
            serialized_parameters = Aggregators.fedavg_aggregate(model_parameters_list)
            SerializationTool.deserialize_model(self._model, serialized_parameters)
            self.round += 1

            self.valid_on_server()

            if self.federated_config.test_rounds:
                if self.round % self.federated_config.log_test_len == 0:
                    result = self.test_on_server()
                    if "test_rounds" not in self.metric_log:
                        self.metric_log["test_rounds"] = {}
                    self.metric_log["test_rounds"][f"round_{self.round}"] = result[
                        self.metric_name
                    ]

            # reset cache cnt
            self.client_buffer_cache = []

            return True  # return True to end this round.
        else:
            return False
        
    def _update_group_model(self, payload, group_id):
        assert len(payload) > 0
        if len(payload) == 1:
            self.client_buffer_cache.append(payload[0].clone())
        else:
            self.client_buffer_cache += payload  # serial trainer

        assert len(self.client_buffer_cache) <= self._group_size

        if len(self.client_buffer_cache) == self._group_size:
            model_parameters_list = self.client_buffer_cache
            self.logger.debug(
                f"Model parameters aggregation, number of aggregation elements {len(model_parameters_list)}"
            )

            # use aggregator
            serialized_parameters = Aggregators.fedavg_aggregate(model_parameters_list)
            self._group_models[group_id] = serialized_parameters

            # reset cache cnt
            self.client_buffer_cache = []

            return True  # return True to end this round.
        else:
            return False

    def _update_global_group_model(self):
        assert len(self._group_models) == self._group_num

        if len(self._group_models) == self._group_num:
            model_parameters_list = self._group_models
            self.logger.debug(
                f"Model parameters aggregation, number of aggregation elements {len(model_parameters_list)}"
            )

            # use aggregator
            serialized_parameters = Aggregators.fedavg_aggregate(model_parameters_list)
            SerializationTool.deserialize_model(self._model, serialized_parameters)
            self.round += 1

            self.valid_on_server()

            if self.federated_config.test_rounds:
                if self.round % self.federated_config.log_test_len == 0:
                    result = self.test_on_server()
                    if "test_rounds" not in self.metric_log:
                        self.metric_log["test_rounds"] = {}
                    self.metric_log["test_rounds"][f"round_{self.round}"] = result[
                        self.metric_name
                    ]

            # reset cache cnt
            self.client_buffer_cache = []

            return True  # return True to end this round.
        else:
            return False

    def downlink_package_of_group(self, gid):
        return [self._group_models[gid]]

    @property
    def client_num_per_round(self):
        return max(1, int(self.sample_ratio * self.client_num_in_total))
    
    @property
    def num_of_group(self):
        return self._group_num
    
    @property
    def downlink_package(self):
        """Property for manager layer. BaseServer manager will call this property when activates clients."""
        return [self.model_parameters]

    @property
    def if_stop(self):
        """
        class:`NetworkManager` keeps monitoring this attribute,
        and it will stop all related processes and threads when ``True`` returned.
        """
        return self.round >= self.global_round

    def valid_on_server(self):

        result = self.eval.test_and_eval(
            model=self._model,
            valid_dl=self.valid_data,
            model_type=self.model_config.model_type,
            model_output_mode=self.model_config.model_output_mode,
        )

        self.on_round_end(result)

    def test_on_server(self):

        SerializationTool.deserialize_model(self._model, self.best_glo_params)

        result = self.eval.test_and_eval(
            model=self._model,
            valid_dl=self.test_data,
            model_type=self.model_config.model_type,
            model_output_mode=self.model_config.model_output_mode,
        )

        self.logger.critical(
            f"task:{self.data_config.task_name}, Setting:{self.metric_log['info']}, "
            f"Test {self.metric_name.upper()}:{result[self.metric_name]:.3f}"
        )

        self.global_test_best_metric = result[self.metric_name]

        return result

    def save_global_model(self):
        torch.save(self.best_glo_params, self.glo_save_file)

    def on_round_end(self, result):
        test_metric, test_loss = result[self.metric_name], result["eval_loss"]

        # TODO hard code
        if self.global_valid_best_metric < test_metric:
            self.global_valid_best_metric = test_metric
            self.best_glo_params = SerializationTool.serialize_model(self._model)

        self.logger.info(
            f"{self.data_config.task_name}-{self.model_config.model_type} "
            f"train with client={self.federated_config.clients_num}_"
            f"alpha={self.federated_config.alpha}_"
            f"epoch={self.training_config.num_train_epochs}_"
            f"seed={self.training_config.seed}_"
            f"comm_round={self.federated_config.rounds}"
        )

        self.logger.debug(
            f"{self.federated_config.fl_algorithm} Eval "
            f"Round:{self.round}, Loss:{test_loss:.3f}, "
            f"Current {self.metric_name}:{test_metric:.3f}, "
            f"Best {self.metric_name}:{self.global_valid_best_metric:.3f}"
        )

        self.metric_log["logs"].append(
            {
                f"round_{self.round}": {
                    "loss": f"{test_loss:.3f}",
                    f"{self.metric.metric_name}": f"{test_metric:.3f}",
                }
            }
        )
        wandb.log({f"{self.metric_name}": test_metric})
        wandb.log({"Loss": test_loss})


class BaseServerManager(ServerManager):
    """Synchronous communication

    BaseServerManager.run()
    setup() main_loop() shut_down()

    """

    def __init__(self, network, handler):
        super(BaseServerManager, self).__init__(network, handler)

        self.logger = registry.get("logger")

    def setup(self):
        self._network.init_network_connection()

        rank_client_id_map = {}

        for rank in range(1, self._network.world_size):
            _, _, content = self._network.recv(src=rank)
            rank_client_id_map[rank] = content[0].item()
        self.coordinator = Coordinator(
            rank_client_id_map, mode="GLOBAL"
        )  # mode='GLOBAL'
        if self._handler is not None:
            self._handler.client_num_in_total = self.coordinator.total

    def main_loop(self):

        while self._handler.if_stop is not True:
            # activate = threading.Thread(target=self.activate_clients)
            # TODO:客户端分组
            self.logger.info(f"########################################################## Round {self._handler.round} start.")
            groups_ids = self._handler.grouped_clients(self._handler.num_of_group) # list of list
            group_step = 2
            local_step = 50
            
            for gid, group_ids in enumerate(groups_ids): # group id list
                # 序列化训练每组模型g_step轮，客户端上进行local_step轮
                self.logger.info(f"########### start group {gid} with client {group_ids}")
                first = True
                for g_step in range(group_step):
                    self.logger.info(f"###### group step = {g_step}, local step = {local_step}")
                    activate = threading.Thread(
                        target=self.activate_clients_in_a_group_with_steps,
                        args=(group_ids, local_step, first, gid)
                    )
                    first = False
                    activate.start()

                    while True:
                        sender_rank, message_code, payload = self._network.recv()

                        if message_code == MessageCode.ParameterUpdate:
                            if self._handler._update_group_model(payload, gid):
                                break
                        else:
                            raise Exception("Unexpected message code {}".format(message_code))
                self.logger.info(f"########### end group {gid} with client {group_ids}")
            self.logger.info(f"Global aggregation.")
            self._handler._update_global_group_model()
            self.logger.info(f"########################################################## Round {self._handler.round - 1} end.")
            

    def shutdown(self):
        """Shutdown stage."""
        self.shutdown_clients()
        super().shutdown()

    def activate_clients(self):

        self.logger.info("BaseClient activation procedure")
        clients_this_round = self._handler.sample_clients()
        rank_dict = self.coordinator.map_id_list(clients_this_round)

        self.logger.info("BaseClient id list: {}".format(clients_this_round))

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package
            id_list = torch.Tensor(values).to(downlink_package[0].dtype)
            self._network.send(
                content=[id_list] + downlink_package,
                message_code=MessageCode.ParameterUpdate,
                dst=rank,
            )

    def activate_clients_in_a_group_with_steps(self, clients_this_round, local_step, first, gid):

        self.logger.info("BaseClient activation procedure")
        rank_dict = self.coordinator.map_id_list(clients_this_round)
        self.logger.info("BaseClient id list: {}".format(clients_this_round))

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package if first else self._handler.downlink_package_of_group(gid)
            values += [local_step]
            id_list_and_step = torch.Tensor(values).to(downlink_package[0].dtype)
            self._network.send(
                content=[id_list_and_step] + downlink_package,
                message_code=MessageCode.ParameterUpdate,
                dst=rank,
            )

    def shutdown_clients(self):
        """Shutdown all clients.

        Send package to each client with :attr:`MessageCode.Exit`.

        Note:
            Communication agreements related: User can overwrite this function to define package
            for exiting information.
        """
        client_list = range(self._handler.client_num_in_total)
        rank_dict = self.coordinator.map_id_list(client_list)

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package
            id_list = torch.Tensor(values).to(downlink_package[0].dtype)
            self._network.send(
                content=[id_list] + downlink_package,
                message_code=MessageCode.Exit,
                dst=rank,
            )

        # wait for client exit feedback
        _, message_code, _ = self._network.recv(src=self._network.world_size - 1)
        assert message_code == MessageCode.Exit
