function [trainedModel, validationRMSE] = trainRegressionModel(trainingData)

% MatLab altındaki Regression Learner App ile otomatik olarak oluşturuldu.

% Altta veri seti hazırlandı.
inputTable = trainingData;
predictorNames = {'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15'};
predictors = inputTable(:, predictorNames);
response = inputTable.price;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]; 

% Veri setinde Kategorik olabilecek özellikler de sayısal olarak verildiğinden hepsi numerik.


% Regression modeli eğitimi ve özellikleri
regressionNeuralNetwork = fitrnet(...
    predictors, ...
    response, ...
    'LayerSizes', [300 300 30], ...
    'Activations', 'relu', ...
    'Lambda', 0, ...
    'IterationLimit', 100, ...
    'Standardize', true);

% Struct oluşturma
predictorExtractionFcn = @(t) t(:, predictorNames);
neuralNetworkPredictFcn = @(x) predict(regressionNeuralNetwork, x);
trainedModel.predictFcn = @(x) neuralNetworkPredictFcn(predictorExtractionFcn(x));

% Struct altındaki ek 
trainedModel.RequiredVariables = {'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15'};
trainedModel.RegressionNeuralNetwork = regressionNeuralNetwork;
trainedModel.About = 'This struct is a trained model exported from Regression Learner R2024a.';
trainedModel.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Tahmin edicileri ve yanıtı çıkarır
% Bu kod, verileri eğitmek için doğru şekle dönüştürür.

inputTable = trainingData;
predictorNames = {'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15'};
predictors = inputTable(:, predictorNames);
response = inputTable.price;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% cross-validation gerçekleştirme (KFold ile)
partitionedModel = crossval(trainedModel.RegressionNeuralNetwork, 'KFold', 10);

% Doğrulama tahminlerini hesaplama
validationPredictions = kfoldPredict(partitionedModel);

% Doğrulama RMSE'sini hesaplama
validationRMSE = sqrt(kfoldLoss(partitionedModel, 'LossFun', 'mse'));
