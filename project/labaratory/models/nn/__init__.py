import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(
            self,
            input_dimension,
            hidden_dimension,
            activation='LeakyReLU',
            output_dimension=1
    ):
        super(NeuralNetwork, self).__init__()
        self.__activation = self.__activate(activation).double()
        self.__input = nn.Linear(
            in_features=input_dimension,
            out_features=hidden_dimension
        ).double()
        self.__output = nn.Linear(
            in_features=hidden_dimension,
            out_features=output_dimension
        ).double()

    def forward(
            self,
            X
    ):
        X = self.__input(X)
        X = self.__activation(X)
        X = self.__output(X)
        return X

    def __activate(
            self,
            func
    ):
        """
        Activation function name
        :type func: str
        """
        # if func is not str:
        #     raise AttributeError(
        #         f'Awaiting str type func name. Received: {type(func)}'
        #     )
        if func == 'LeakyReLU':
            return nn.LeakyReLU()
        if func == 'ReLU':
            return nn.ReLU()


# def make_prediction(
#         X_train,
#         y_train,
#         # X_test,
#         params
# ):
#     """
#     Predictor for Neural Network
#     :type X_train: numpy.ndarray
#     :type y_train: numpy.ndarray
#     :type X_test: numpy.ndarray
#     :type params: dict
#     """
#     train_loader, validation_loader, test_loader = prepare_data(
#         X=X_train,
#         y=y_train,
#         batch_size=params['batch_size'],
#         test_size=params['test_size'],
#         valid_size=params['validation_size'],
#     )
#     nn = model.NeuralNetwork(
#         input_dimension=len(X_train),
#         hidden_dimension=params['hidden_dimension'],
#         output_dimension=len(y_train)
#     )
#     optimizer = torch.optim.SGD(
#         params=nn.parameters(),
#         lr=params['alpha']
#     )
#     criterion = torch.nn.MSELoss()
#     metric = mean_squared_error
#     stats = model_validation.train_model(
#         model=nn,
#         criterion=criterion,
#         optimizer=optimizer,
#         train_loader=train_loader,
#         num_epochs=params['num_epochs'],
#         metric=metric,
#         device=params['cpu'],
#         print_step=params['print_step']
#     )
#     print(stats)
#     validation_pred, validation_metrics = model_validation.test_model(
#         model=nn,
#         loader=validation_loader,
#         metric=metric,
#         device=params['cpu'],
#         isValidation=True
#     )
#     print(validation_metrics)
#     test_pred, test_metrics = model_validation.test_model(
#         model=nn,
#         loader=test_loader,
#         metric=metric,
#         device=params['cpu'],
#         isValidation=False
#     )
#     print(test_metrics)
#     # return nn.forward(torch.tensor(X_test))
