import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import argparse
from collections import defaultdict,OrderedDict
import random
from PIL import Image
from Model_Wrapper_reid import ResNet_Loader
import os
import glob

import numpy as np

image_ext = ['jpg', 'jpeg', 'png', 'webp']

if __name__ == '__main__':


    args = parser.parse_args()


    # Load ReID Model
    print('loading model....')
    model = ResNet_Loader(args.load_ckpt,args.n_layer,output_color=False,batch_size=int(args.batch_size))

    # Load query images locally
    if os.path.isdir(args.query):
        query_names = []
        ls = os.listdir(args.query)
        for query_name in sorted(ls):
            ext = query_name[query_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                query_names.append(os.path.join(args.query, query_name))
    else:
        query_names = [args.query]

    q_features = model.q_inference(query_names)

    # Run CenterTrack and output bbox



    with open(args.gallery,'r') as f:
        gallery_txt = [q.strip() for q in f.readlines()]
        gallery_txt = gallery_txt[1:]
    print('inferencing q_features')
    #print(query_txt)
    # q_features = model.inference(query_txt)

    print('inferencing g_features')
    g_features = model.g_inference(gallery_txt)

    q_features = nn.functional.normalize(q_features,dim=1).cuda()
    g_features = nn.functional.normalize(g_features,dim=1).transpose(0,1).cuda()
    
    print('compute distance')
    SimMat = torch.mm(q_features,g_features)
    SimMat = SimMat.cpu().transpose(0,1)

    print("SimMat's Size: ", SimMat.size())
    
    SimMat = SimMat.numpy()
    print("SimMat: ", SimMat)
    import scipy.io as sio
    sio.savemat(args.dis_mat,{'dist_CNN':SimMat})
    
