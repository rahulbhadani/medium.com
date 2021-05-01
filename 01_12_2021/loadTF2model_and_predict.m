layer = importKerasLayers('/home/ivory/VersionControl/medium.com/01_12_2021/dnn_model.h5', 'OutputLayerType','regression')
lgraph = layerGraph(layer);
layer_last = regressionLayer('Name','routput')
lgraph = replaceLayer(lgraph,findPlaceholderLayers(layer)',layer_last)
net = assembleNetwork(lgraph)
 net.predict([1, 2])