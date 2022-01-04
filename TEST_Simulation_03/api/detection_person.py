
# https://microsoft.github.io/AirSim/object_detection/
# https://data-make.tistory.com/170 폴더 만들기
#import setup_path 
# from _typeshed import Self
import airsim
import cv2
import numpy as np 
import pprint
import time
import os.path
from datetime import datetime
import json
from json import JSONEncoder
import time
from time import strftime # 시간복잡도 확인
import msvcrt
import msgpackrpc
import asyncio



class Api:



    # subclass JSONEncoder
    class DetectionEncoder(JSONEncoder):
            def default(self, o):
                return o.__dict__


    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)


    #4개의 좌표 받아서 욜로 라벨형태로 반환함수
    def polygon2yolo(x1,x2,y1,y2):
    
        yolo_list = []
    
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = (x2 - x1)
        h = (y2 - y1)
    
        # 768x452
        norm_cx = float(cx / 768)
        norm_cy = float(cy / 452)
        norm_w = float(w / 768)
        norm_h = float(h / 452)

        # 4K
        #norm_cx = float(cx / 3840)
        #norm_cy = float(cy / 2160)
        #norm_w = float(w / 3840)
        #norm_h = float(h / 2160)
    
        yolo_list.extend([norm_cx, norm_cy, norm_w, norm_h])
    
        return yolo_list
        
    
    # 이미지 저장
    def save_scenery_orignal(image, filenames):
        # print("save_scenery_original")
        #21.11.18변경 전/ 경로 C:\Users\sim\Documents\AirSim\datasets  
        #filepath =os.path.expanduser('~')+'\Documents\\AirSim\\datasets\\'
        #21.11.18변경/ 경로 C:\Users\sim\Documents\Sim2data\datasets 
        filepath =os.path.expanduser('~')+'\Documents\\Sim2data\\datasets\\'
        Api.createFolder(filepath)
        cv2.imwrite(filepath + filenames + '.jpg', image)

    # 라벨 데이터 저장
    def save_yolo_label(list_cordinate, filenames):
    
        #21.11.18변경 전/경로 C:\Users\sim\Documents\AirSim\datasets 
        #filepath =os.path.expanduser('~')+'\Documents\\AirSim\\datasets\\'
        #21.11.18변경/ 경로 C:\Users\sim\Documents\Sim2data\datasets #1118변경후
        filepath =os.path.expanduser('~')+'\Documents\\Sim2data\\datasets\\'
        st=''
        dict_ = {'person':'0','PAD':'1','mark':'2'}
        for cordinate in list_cordinate:
            
            yolo_coord = Api.polygon2yolo(cordinate[1],cordinate[2],cordinate[3],cordinate[4]) #욜로 변환
            yolo_coord = list(map(str,yolo_coord))
            st +=  dict_[cordinate[0].split('_')[0]] +' '+yolo_coord[0]+' '+yolo_coord[1]+' '+yolo_coord[2]+' '+yolo_coord[3]+'\n'
        Api.createFolder(filepath)
        print("yolo라벨:"+st)
        
        with open(filepath+filenames+'.txt', 'w', encoding='utf-8') as f:
           f.writelines(st[:-1])
        #print("save_yolo_label")

    # BBox표시된 이미지 저장  
    def save_scenery_bbox(image, filenames):
        #print("save_scenery_bbox")
        #경로 C:\Users\sim\Documents\AirSim\scenery_bbox #211118변경전
        #filepath =os.path.expanduser('~')+'\Documents\\AirSim\\datasets_bbox\\'

        #경로 C:\Users\sim\Documents\Sim2data\datasets #211118변경후
        filepath =os.path.expanduser('~')+'\Documents\\Sim2data\\datasets_bbox\\'
        Api.createFolder(filepath)
    
        cv2.imwrite(filepath + filenames + '.jpg', image)



    def __init__(self):
        self.client = airsim.VehicleClient()
        #self.client.confirmConnection()
        self.camera_name_0 ="0"
        #self.camera_name_1 ="1"
        self.image_type = airsim.ImageType.Scene
        self.object_0 = "Person*"
        self.object_1 = "PAD*"
        self.object_2 = "mark*"
        #self.setting()
        pass

    def setting(self,camera_name = "0", image_type = airsim.ImageType.Scene, object = "Person*",set=0):
        
        
        self.camera_name = camera_name
        self.image_type = image_type
        self.object_0 = "Person*"
        self.object_1 = "PAD*"
        self.client.simSetDetectionFilterRadius(self.camera_name, self.image_type, 800 * 100) 
        self.client.simAddDetectionFilterMeshName(self.camera_name, self.image_type, self.object_0)
        self.client.simAddDetectionFilterMeshName(self.camera_name, self.image_type, self.object_1) 
        self.client.simAddDetectionFilterMeshName(self.camera_name, self.image_type, self.object_2) 
       




    def confirmConnection(self):

        try:
            self.client = airsim.VehicleClient()
            self.client.confirmConnection()
            self.setting()
            #print(self.client.confirmConnection())
            return True
        except msgpackrpc.error.TransportError as e:
            print("not contected")
            return False


    async def get_img(self, image_type):
        await asyncio.sleep(0.1)
        print(datetime.now())
        return self.client.simGetImages([airsim.ImageRequest(0,image_type),airsim.ImageRequest(0,image_type)])


    async def test_detection(self):
        # self.confirmConnection()
        images = await asyncio.gather(self.get_img(self.image_type))
        pngs = [cv2.imdecode(airsim.string_to_uint8_array(i), cv2.IMREAD_UNCHANGED) for i in images]
        pngs = [cv2.resize(i, dsize=(550, 420)) for i in pngs]
        png = cv2.vconcat(pngs)
        Api.save_scenery_bbox(png,datetime.now().strftime("%Y%m%d-%H%M%S") +'_tes')
        cv2.imshow("Sim2Real", png)
        cv2.waitKey(1)

        # images = self.client.simGetImages([airsim.ImageRequest(0,self.image_type),airsim.ImageRequest(0,self.image_type)])
        # filepath =os.path.expanduser('~')+'\Documents\\Sim2data\\datasets\\'
        # filename = filepath + datetime.now().strftime("%Y%m%d-%H%M%S") + '_test'
        # pngs = []
        # for i, response in enumerate(images):
        #     if response.pixels_as_float:
        #         print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
        #         airsim.write_pfm(os.path.normpath(filename + str(i) + '.pfm'), airsim.get_pfm_array(response))
        #     else:
        #         print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
        #         # cv2.imshow('img',response.image_data_uint8)
        #         cv2.imshow('img',cv2.imdecode(np.fromstring(response.image_data_uint8,np.uint8), cv2.IMREAD_UNCHANGED))
        #         cv2.waitKey(1)
        #         # cv2.imshow('img',cv2.imdecode(airsim.string_to_uint8_array(response.image_data_uint8), cv2.IMREAD_UNCHANGED))
        #         airsim.write_file(os.path.normpath(filename+ str(i) + '.png'), response.image_data_uint8)
        # print(images[0].type)

    def exeDetection(self):
        self.confirmConnection() # 연결확인

        start_time = time.time() # 시작시간
        filenames = datetime.now().strftime("%Y%m%d-%H%M%S") +'_syj'

        rawImage = self.client.simGetImage(self.camera_name, self.image_type)
        if not rawImage:
            print("이미지없음")
        else:
            people = self.client.simGetDetections(self.camera_name, self.image_type)
            test_png = [self.client.simGetImage(self.camera_name,self.image_type), self.client.simGetImage(self.camera_name,self.image_type)]
            test_png = [cv2.imdecode(airsim.string_to_uint8_array(i), cv2.IMREAD_UNCHANGED) for i in test_png]
            png = cv2.imdecode(airsim.string_to_uint8_array(self.client.simGetImage(self.camera_name, self.image_type)), cv2.IMREAD_UNCHANGED)
            png2 = cv2.imdecode(airsim.string_to_uint8_array(self.client.simGetImage(self.camera_name, airsim.ImageType.Infrared)), cv2.IMREAD_UNCHANGED)
            # print(people)
            list_cordinate = []
            if people:
                Api.save_scenery_orignal(png, filenames) #원본이미지 저장
                for i,person in enumerate(people):

                    personJSONData = json.dumps(person, indent=4, cls=Api.DetectionEncoder) # str타입

                    # cv2.rectangle => bbox
                    # cv2.putText=> 식별자 이름
                    cv2.rectangle(png,(int(person.box2D.min.x_val),int(person.box2D.min.y_val)),(int(person.box2D.max.x_val),int(person.box2D.max.y_val)),(255,0,0),2)
                    print("이미지 좌표:{},{},{},{}".format(person.box2D.min.x_val,person.box2D.min.y_val,person.box2D.max.x_val,person.box2D.max.y_val))
                    cv2.putText(png, person.name , (int(person.box2D.min.x_val),int(person.box2D.min.y_val - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (36,255,12))
 
                
                    list_cordinate.append([]) #이중리스트 만들기
                    list_cordinate[i].append(person.name)
                    list_cordinate[i].append(person.box2D.min.x_val)
                    list_cordinate[i].append(person.box2D.max.x_val)
                    list_cordinate[i].append(person.box2D.min.y_val)
                    list_cordinate[i].append(person.box2D.max.y_val)


                Api.save_yolo_label(list_cordinate,filenames ) # 라벨 데이터(txt) 저장
                png = cv2.resize(png, dsize=(550, 420))
                png2 = cv2.resize(png2, dsize=(550, 420))
                png3 = [cv2.resize(i, dsize=(550, 420)) for i in test_png]
                # png = cv2.vconcat(png3)
                Api.save_scenery_bbox(png, filenames) #Box 표시된 이미지 저장
                cv2.imshow("Sim2Real", png)
                cv2.waitKey(1)
                self.client.simClearDetectionMeshNames(self.camera_name, self.image_type)
                # if cv2.waitKey(100) & 0xFF == ord('q'):
                #     print("원래는 continue")
                # elif cv2.waitKey(100) & 0xFF == ord('c'):
                # elif cv2.waitKey(100) & 0xFF == ord('a'):
                #     self.client.simAddDetectionFilterMeshName(self.camera_name, self.image_type, self.object_0)
                #     self.client.simAddDetectionFilterMeshName(self.camera_name, self.image_type, self.object_1)
                #cv2.destroyAllWindows()
                     
            else:
                 print("사람 없음")
      
            
        print("작업시간:{}초".format(time.time() - start_time))  # (끝시간 - 시작 시간) 출력
        
