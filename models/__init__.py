# -*- coding: utf-8 -*-


#from models.match.keras.RealNN import RealNN
#from models.match.keras.QDNN import QDNN
#from models.match.keras.ComplexNN import ComplexNN
##from .QDNNAblation import QDNNAblation
#from models.match.keras.LocalMixtureNN import LocalMixtureNN

def setup(opt):
    
    if opt.dataset_type == 'qa':
#            from models.match.pytorch.RealNN import RealNN
#            from models.match.pytorch.QDNN import QDNN
#            from models.match.pytorch.ComplexNN import ComplexNN
#            from models.match.pytorch.LocalMixtureNN import LocalMixtureNN
        print('None')
        
    elif opt.dataset_type == 'classification':
#            from models.classification.pytorch.RealNN import RealNN
        from models.classification.QDNN import QDNN
        from models.classification.MLLM import MLLM
        from models.classification.QTNET import QTNET
#            from models.classification.pytorch.ComplexNN import ComplexNN
#            from models.classification.pytorch.QDNNAblation import QDNNAblation
        from models.classification.LocalMixtureNN import LocalMixtureNN
    
    
    print("network type: " + opt.network_type)
    if opt.network_type == "real":
        model = RealNN(opt)
    elif opt.network_type == "qdnn":
        model = QDNN(opt)
    elif opt.network_type == "complex":
        model = ComplexNN(opt)
    elif opt.network_type == "local_mixture":
        model = LocalMixtureNN(opt)
    elif opt.network_type == "mllm":
        model = MLLM(opt)
    elif opt.network_type == "qtnet":
        model = QTNET(opt)
        
    elif opt.network_type == "bert":
        from models.representation.keras.BertFasttext import BERTFastext
        model = BERTFastext(opt)
        
    elif opt.network_type == "ablation":
        print("run ablation")
        model = QDNNAblation(opt)
    else:
        raise Exception("model not supported: {}".format(opt.network_type))
    return model
