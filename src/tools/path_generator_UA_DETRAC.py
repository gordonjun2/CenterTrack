import os


f = open('path_list.txt','w+')


xmls = os.listdir('/home/coffeemix/Desktop/Gordon/CenterTrack/data/UA-DETRAC/DETRAC-test-data/DETRAC-Test-Annotations-XML')

i = 0
for xml in xmls:
    f.write('/home/coffeemix/Desktop/Gordon/CenterTrack/data/UA-DETRAC/DETRAC-test-data/DETRAC-Test-Annotations-XML/' + str(xml) + '\n')
    i += 1
    print("Count: ", i)
f.close()
