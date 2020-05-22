from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import json
import copy
import numpy as np
from opts import opts
from detector import Detector

from AIC2018_iamai.ReID.ReID_CNN.Model_Wrapper_reid import ResNet_Loader
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import numpy as np
from utils.drawer import Drawer
import time

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']
drawer = Drawer()
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

def euclidean_distance(qf, gf):
  gf = gf.transpose(0,1)
  m = qf.shape[0]
  n = gf.shape[0]
  dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
  dist_mat.addmm_(1, -2, qf, gf.t())
  return dist_mat.cpu().numpy()


def cosine_similarity(qf, gf):
  gf = gf.transpose(0,1)
  epsilon = 0.00001
  dist_mat = qf.mm(gf.t())
  qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
  gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
  qg_normdot = qf_norm.mm(gf_norm.t())

  dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
  dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
  dist_mat = np.arccos(dist_mat)
  return dist_mat

def demo(opt, reid_model = None, q_features = None):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  print('loading CenterTrack....')
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    out = None
    out_name = opt.demo[opt.demo.rfind('/') + 1:]
    if opt.save_video:
      fourcc = cv2.VideoWriter_fourcc(*'XVID')
      out = cv2.VideoWriter('../results/{}.mp4'.format(
        opt.exp_id + '_' + out_name),fourcc, opt.save_framerate, (
          opt.video_w, opt.video_h))
    detector.pause = False
    cnt = 0
    results = {}
    if opt.load_results != '':
      load_results = json.load(open(opt.load_results, 'r'))
    while True:
        cnt += 1
        check, img = cam.read()
        if check ==  True:
          if opt.resize_video:
            try:
              img = cv2.resize(img, (opt.video_w, opt.video_h))
            except:
              print('FINISH!')
              save_and_exit(opt, out, results, out_name)
          if cnt < opt.skip_first:
            continue
          # try:
          #   cv2.imshow('input', img)
          # except:
          #   print('FINISH!')
          #   save_and_exit(opt, out, results, out_name)
          input_meta = {'pre_dets': []}
          img_id_str = '{}'.format(cnt)
          if opt.load_results:
            input_meta['cur_dets'] = load_results[img_id_str] \
              if img_id_str in load_results else []
            if cnt == 1:
              input_meta['pre_dets'] = load_results[img_id_str] \
                if img_id_str in load_results else []
          start_time = time.time()
          ret = detector.run(img, input_meta)
          print(ret)
          # time_str = 'frame {} |'.format(cnt)
          # for stat in time_stats:
          #   time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
          # results[cnt] = ret['results']
          # print(time_str)
          if opt.reid and len(ret['results']) != 0:
            gallery_list = []
            # track_dict = {}
            for i in range(len(ret['results'])):
              l, t, r, b = ret['results'][i]['bbox']
              # track[i] = ret['results'][i]['tracking_id']

              if l < 0:
                l = 0
              if r > img.shape[1]:
                r = img.shape[1]
              if t < 0:
                t = 0
              if b > img.shape[0]:
                b = img.shape[0]

              cropped_bbox = img[int(t):int(b), int(l):int(r)]
              gallery_list.append(cropped_bbox)

            print('inferencing g_features')
            g_features = reid_model.g_inference(gallery_list)

            q_features = nn.functional.normalize(q_features,dim=1).cuda()
            g_features = nn.functional.normalize(g_features,dim=1).transpose(0,1).cuda()
            
            print('compute distance')

            if opt.dist == 'cosine':
              dist = cosine_similarity(q_features, g_features)
            else:
              dist = euclidean_distance(q_features, g_features)

            #print("Distance's Size: ", dist.shape)
            
            dist = dist.transpose()
            query_len = dist.shape[1]

            end_time = time.time()
            total_time = end_time - start_time
            print("Total Time:" + str(total_time) + "s")

            best_index = []

            ##### Multiple Query Images of the same vehicle #####
            if query_len > 1 and opt.single_query:
              summed = np.sum(dist, axis = 1)
              best_index.append(np.argmin(summed, axis = 0))
              final_frame = drawer.draw_dets_video(img, ret, total_time, opt.reid, summed, best_index, query_len)
              cv2.imshow("Frame", final_frame)

            ##### Multiple Query Images of different vehicles #####
            elif query_len > 1 and not opt.single_query:
              for i in range(query_len):
                query_column = dist[:, i]
                best_index.append(np.argmin(query_column, axis = 0))
              final_frame = drawer.draw_dets_video(img, ret, total_time, opt.reid, dist, best_index, query_len)
              cv2.imshow("Frame", final_frame)

            ##### Single Query Image #####
            elif query_len == 1:
              best_index.append(np.argmin(dist, axis = 0)[0])
              final_frame = drawer.draw_dets_video(img, ret, total_time, opt.reid, dist, best_index, query_len)
              cv2.imshow("Frame", final_frame)
          
          else:
            end_time = time.time()
            total_time = end_time - start_time
            print("Total Time:" + str(total_time) + "s")
            final_frame = drawer.draw_dets_video(img, ret, total_time, opt.reid)
            cv2.imshow("Frame", final_frame)

          if opt.save_video:
            out.write(ret['generic'])
          if cv2.waitKey(1) == 27:
            print('EXIT!')
            save_and_exit(opt, out, results, out_name)
            return  # esc to quit
        else:
          print("Video has ended!")
          break
    save_and_exit(opt, out, results)
  else:
    # Demo on images, currently does not support tracking
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    for (image_name) in image_names:
      ret = detector.run(image_name)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)

def save_and_exit(opt, out=None, results=None, out_name=''):
  print('results')
  if opt.save_results and (results is not None):
    save_dir =  '../results/{}_results.json'.format(opt.exp_id + '_' + out_name)
    print('saving results to', save_dir)
    json.dump(_to_list(copy.deepcopy(results)), 
              open(save_dir, 'w'))
  if opt.save_video and out is not None:
    out.release()
  import sys
  sys.exit(0)

def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results

if __name__ == '__main__':
  opt = opts().init()
  
  if opt.reid:
    # Load ReID Model
    if opt.reid_model == "resnet18":
      load_ckpt = './AIC2018_iamai/ReID/ReID_CNN/models/model_280_base_resnet18.ckpt'
      n_layer = 18
    else:
      load_ckpt = './AIC2018_iamai/ReID/ReID_CNN/models/model_880_base_resnet50.ckpt'
      n_layer = 50

    print('loading re-id model....')
    reid_model = ResNet_Loader(load_ckpt, n_layer,output_color=False,batch_size=int(opt.reid_batch_size))
    
    # Load query images locally
    if os.path.isdir(opt.query):
      query_names = []
      ls = os.listdir(opt.query)
      for query_name in sorted(ls):
        ext = query_name[query_name.rfind('.') + 1:].lower()
        if ext in image_ext:
          query_names.append(os.path.join(opt.query, query_name))
    else:
        query_names = [opts.query]

    print('inferencing q_features')
    q_features = reid_model.q_inference(query_names)
    #print(q_features)

    demo(opt, reid_model, q_features)
  
  else:
    demo(opt)
