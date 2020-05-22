from xml.dom import minidom
import os
import argparse



if __name__ == '__main__':
  # Argparse
    parser = argparse.ArgumentParser(description='Database generator for VRIC dataset')
    parser.add_argument('--input_train_txt', help='the output txt file listing all imgs to database and its label')
    parser.add_argument('--input_query_txt', help='the output txt file listing all imgs to database and its label')
    parser.add_argument('--input_gallery_txt', help='the output txt file listing all imgs to database and its label')
    parser.add_argument('--output_train_txt', help='the output txt file listing all imgs to database and its label')
    parser.add_argument('--output_query_txt', help='the output txt file listing all imgs to database and its label')
    parser.add_argument('--output_gallery_txt', help='the output txt file listing all imgs to database and its label')
    args = parser.parse_args()

    output_train = open(args.output_train_txt,'w')
    input_train = open(args.input_train_txt, "r")

    train_id = []

    for line in input_train:
        line = line.split()
        train_id.append(line[1])

    train_id = list(dict.fromkeys(train_id))

    print("No. of unique cars: ", len(train_id))

    id_dict = {}

    count = 1
    for keys in train_id:
        id_dict[keys] = str(count)
        count = count + 1

    input_train = open(args.input_train_txt, "r")

    output_train.write('img_path id\n')
    for line in input_train:
        line = line.split()
        line = line[:-1]
        line[1] = id_dict[line[1]]
        line = ' '.join([str(elem) for elem in line]) 
        output_train.write(os.path.join(os.getcwd(), "datasets", "VRIC", "train_images", line)+'\n')
    output_train.close()

    output_query = open(args.output_query_txt,'w')
    input_query = open(args.input_query_txt, "r")

    output_query.write('img_path id\n')
    for line in input_query:
        line = line.split()
        line = line[:-1]
        line = ' '.join([str(elem) for elem in line]) 
        output_query.write(os.path.join(os.getcwd(), "datasets", "VRIC", "probe_images", line)+'\n')
    output_query.close()

    output_gallery = open(args.output_gallery_txt,'w')
    input_gallery = open(args.input_gallery_txt, "r")

    output_gallery.write('img_path id\n')
    for line in input_gallery:
        line = line.split()
        line = line[:-1]
        line = ' '.join([str(elem) for elem in line]) 
        output_gallery.write(os.path.join(os.getcwd(), "datasets", "VRIC", "gallery_images", line)+'\n')
    output_gallery.close()

    # for line in input_train:
    #     line = line.split()
    #     line[0] = os.getcwd() + "/datasets/VRIC/train_images/" + line[0]
    #     print(line)
        

    # img_dir = args.img_dir

    # txt_file = open(args.train_txt,'w')
    
    # img_list = []
    # V_ID_list = []
    # colorID_list = []
    # typeID_list = []
    # V_ID_dict = {}
    # count = 1
    # for s in itemlist:
    #     img_name = s.attributes['imageName'].value
    #     img_list.append(os.path.join(img_dir,img_name))

    #     V_ID = s.attributes['vehicleID'].value
    #     if V_ID not in V_ID_dict:
    #         V_ID_dict[V_ID]=count
    #         count +=1 
    #     V_ID_list.append(V_ID)
    #     colorID_list.append(s.attributes['colorID'].value)
    #     typeID_list.append(s.attributes['typeID'].value)

    # txt_file.write('img_path id color type\n')
    # for i in range(len(img_list)):
    #     img_path = img_list[i]
    #     V_ID = str(V_ID_dict[V_ID_list[i]])
    #     colorID = colorID_list[i]
    #     typeID = typeID_list[i]
    #     txt_file.write(img_path+' '+V_ID+' '+colorID+' '+typeID+'\n')
    # txt_file.close()
    # xmlfile.close()

    # #query 
    # img_list = os.listdir(args.query_dir)
    # img_list.sort()
    # file = open(args.query_txt,'w')
    # file.write('img_path\n')
    # for i in img_list:
    #     file.write(os.path.join(args.query_dir,i)+'\n')
    # file.close()

    # #gallery
    # img_list = os.listdir(args.gallery_dir)
    # img_list.sort()
    # file = open(args.gallery_txt,'w')
    # file.write('img_path veh_id cam_id\n')
    # for i in img_list:
    #     file.write(os.path.join(args.gallery_dir,i)+'\n')
    # file.close()



