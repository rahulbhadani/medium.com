function accel = loadTF_and_Predict(modelpath, v_in, yaw_rate_in)

    layers = importTensorFlowLayers(modelpath);
    layer_last = regressionLayer('Name','routput');
    lgraph = replaceLayer(layers,findPlaceholderLayers(layers).Name,layer_last);
    net = assembleNetwork(lgraph);
    accel = net.predict([v_in, yaw_rate_in]);
end
