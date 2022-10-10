# -*- coding: UTF-8 -*-
#!/usr/bin/python2
# import ctypes
# libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import rospy, sys
from std_msgs.msg import Bool
import math
from geometry_msgs.msg import PoseStamped, Pose
# from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
import threading
import time
from geometry_msgs.msg import Twist
import matplotlib.pyplot as plt
#from scipy.interpolate import spline
import ur_kinematics as kmtic
import rtde_receive
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import cv2.aruco as aruco

from ctypes import *
import random
import os
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
from scipy.spatial.transform import Rotation

import eye_in_hand



class Image_converter:
    def __init__(self):
 
        self.needle_lost_times = 0
        self.find_needle = False
        self.first_distence = 0
        self.ur_kmtic = kmtic.Kinematic()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3,1600)
        self.cap.set(4,1200)
        self.center_point = [1010,606]#[453,266]
        # self.center_point = [367,267]
        self.threading_flag, self.frame_yolo = self.cap.read()
        self.frame_QR_after=self.frame_yolo
        print('cam state :',self.threading_flag)
        self.second_flag_time = 0
        
        # self.dist=np.array(([[0.082377 ,-0.089929, 0.004516 ,-0.006607 ,0.000000]]))
        # self.mtx=np.array([[504.575364 ,0.000000, 307.274235],
        #                   [ 0.000000 ,506.176717 ,247.898462],
        #                   [  0.,           0.,           1.        ]])
        self.dist=np.array(([[0.070725, -0.077215, -0.005932, 0.009241, 0.000000]]))
        self.mtx=np.array([[1243.873830, 0.000000 ,832.799544],
                          [ 0.000000, 1241.795669, 572.282181],
                          [  0.,           0.,           1.        ]])
        show_plt_flag = 0
        show_plt_flag_dis = 0

        # self.eye2hand=[[ 0.25266084, -0.19005527 , 0.94870517 ],
        #                 [ 0.13122078 ,-0.96472812 ,-0.22821212  ],
        #                 [ 0.95861546 , 0.1821501 , -0.21880982 ]
        #                 ]
        self.eye2hand=[[ 0.98716204, -0.01173908,  0.15928997 ],
                        [-0.00212113 , 0.99624397 , 0.08656468  ],
                        [-0.15970787, -0.08579124,  0.98342934 ],
                       ]
        self.eye2hand_T=[[ 0.98716204, -0.01173908,  0.15928997 , 0.13499538],
                        [-0.00212113 , 0.99624397 , 0.08656468 ,-0.0017232 ],
                        [-0.15970787, -0.08579124,  0.98342934 ,-0.19350835],
                        [ 0.        ,  0.       ,   0.        ,  1.        ]]


        # self.eye2hand_T=[[ 0.25266084, -0.19005527 , 0.94870517 , 0.05319715],
        #                     [ 0.13122078 ,-0.96472812 ,-0.22821212  ,0.0933525 ],
        #                     [ 0.95861546 , 0.1821501 , -0.21880982 , 0.03892601],
        #                     [ 0.      ,    0.    ,     0.   ,       1.        ]]

        self.second_flag=False
        self.last_print_state_ttsp = False
        self.last_print_state_btfp = False
        self.start_T = [0]*16
        self.vector_x = 0
        self.now_T = []
  
        rospy.init_node('ur_image_node', anonymous=True)
        self.twist_pub_first = rospy.Publisher('/dhq/image/twist_first', Twist, queue_size=1)
        self.part_pub = rospy.Publisher('/dhq/first_part', Bool, queue_size=1)
        self.twist_pub_second = rospy.Publisher('/dhq/image/twist_second', Twist, queue_size=1)
        self.back_flag_pub = rospy.Publisher('/dhq/back_part', Bool, queue_size=1)

        self.rtde = rtde_receive.RTDEReceiveInterface("192.168.2.233")

        self.args = self.parser()
        self.network, self.class_names, self.class_colors = darknet.load_network(
                self.args.config_file,
                self.args.data_file,
                self.args.weights,
                batch_size=1
            )
        self.darknet_width = darknet.network_width(self.network)
        self.darknet_height = darknet.network_height(self.network)
        self.frame_queue = Queue()
        self.QRframe_queue = Queue()
        self.darknet_image_queue = Queue(maxsize=1)
        self.detections_queue = Queue(maxsize=1)
        self.fps_queue = Queue(maxsize=1)

    
        
        
        self.END_TIME = 5
        self.second_part_end_flag = False
        self.end_time = 0
      
        self.last_print_state_fw = False
        self.last_print_state_end = False
        self.last_print_state_bk = False
        self.needle_lost_time_state = False
        self.depth = 0
    
        # self.T = []
        # self.T_last = []
        # self.U_last = 0
        # self.V_last = 0
        # self.distence_u = []
        # self.distence_v = []
        self.distence_2 = []
        self.distence_now = []
        self.time_list = []
        self.start_time = time.time()
        self.T_list = []
        self.U_V_list = [[],[]]
        self.hand = []
        self.camera = []
        # self.P_list = []
        # self.distence_NL = []
        self.dark_Thread()
        self.shutdown_event = threading.Event()
        # try:
        #     while True:
        #         prev_time = time.time()
                
        #         ret , fream = self.cap.read()
        #         if not ret :
        #             print("false in reading cam")
        #         # self.video_capture(self.frame_queue, self.darknet_image_queue, fream)
        #         # Thread(target=self.video_capture, args=(self.frame_queue, self.darknet_image_queue)).start()
        #         self.callback(fream)
        #         # Visualize count down
        #         # if time.time() -T_start > 5 + 5:
        #         #     pipeline.stop()
        #         #     break
        #         #cv2.imshow('COLOR IMAGE',c)
        #         # press q to quit the program
        #         fps = int(1/(time.time() - prev_time))
        #         if cv2.waitKey(fps) == ord('q'):
        #             self.threading_flag = False
        #             break
        # except KeyboardInterrupt:
        #         self.threading_flag = False
        # cv2.destroyAllWindows()
        # self.cap.release()
        
        # if (show_plt_flag_dis):
   
        #   # ax1 = fig.add_subplot(2,1,1) # 画2行1列个图形的第1个
        #   # plt.xlim((0, len(self.time_list)))
        #   plt.figure(0)
        # #   plt.plot(self.time_list, self.distence_u,label='U',color='g')
        # #   plt.plot(self.time_list, self.distence_now ,label='Real',color='r')
        # #   plt.plot(self.time_list, self.distence_v ,label='V',color='y')
        # #   plt.ylabel('Distance (m)')
        # #   plt.xlabel('time (s)')
        # #   plt.legend(loc='upper right')
        # #   plt.legend()
        # #   y_min,y_max =plt.ylim()

        # #   plt.figure(1)
        #   plt.plot(self.time_list, self.distence_2,label='MRE',color='b')
        #   plt.plot(self.time_list, self.distence_now ,label='Real',color='r')
        #   plt.ylabel('Distance (s)')
        #   plt.xlabel('time (s)')
        #   plt.legend(loc='upper right')
        # #   plt.ylim(y_min,y_max)
        #   plt.legend()
        #   print("erros",np.sum(abs(np.array(self.distence_now)-np.array(self.distence_2)))/len(self.distence_2) )
        if (show_plt_flag):
          time_sm = np.array(self.time_list)
          fig = plt.figure()#plt.figure(0)
          ax1 = fig.add_subplot(2,1,1) # 画2行1列个图形的第1个
          plt.xlim((0, len(self.time_list)))
          plt.ylim((min(self.feedback_list_x)-0.05, max(self.feedback_list_x)+0.05))
          plt.plot(self.time_list, self.setpoint_list,label='setpoint')
          plt.plot(self.time_list, self.feedback_list_x,label='x_after_PID')
          plt.plot(self.time_list, self.list_x,label='x')
          plt.ylabel('x')
          plt.legend()

          ax2 = fig.add_subplot(2,1,2) # 画2行1列个图形的第2个
          plt.xlim((0, len(self.time_list)))
          plt.ylim((min(self.feedback_list_z)-0.05, max(self.feedback_list_z)+0.05))
          plt.plot(self.time_list, self.setpoint_list,label='setpoint')
          plt.plot(self.time_list, self.feedback_list_z,label='z_after_PID')
          plt.plot(self.time_list, self.list_z,label='z')
          plt.xlabel('time (s)')
          plt.ylabel('z')
          plt.legend()
          #plt.grid(True)
          plt.show()

    def get_frame(self):

      while self.threading_flag:
          # print('**************************************************')
          ret, frame = self.cap.read()
          #self.frame_QR = frame.copy()
          if ret:
            self.QRframe_queue.put(frame.copy())
            # self.video_capture(frame,self.frame_queue, self.darknet_image_queue)
            # if self.yolo_cap:
              # self.yolo_cap = 0
            # Thread(target=self.video_capture, args=(frame,self.frame_queue,self.darknet_image_queue)).start()
          else:
              break
          if not self.threading_flag:
            time.sleep(3)
            print('get_frame end')
            break
  
   


    def get_cur_T(self):
            self.cur_joint = self.rtde.getActualQ()
            T=self.ur_kmtic.Forward(self.cur_joint)
            return T
    def get_distence_list(self,Un,Vn,distence,T_temp):
        if Un+Vn == 0:
          return 0
        self.T_list.append(T_temp)
        self.U_V_list[0].append(Un)
        self.U_V_list[1].append(Vn)

  
        if len(self.T_list)==1:
          return 0
        if len(self.T_list)>6:
          del self.T_list[0]
          del self.U_V_list[0][0]
          del self.U_V_list[1][0]
    
        distence_all = []
        k_dis = []
        dist_pose_all = 0
        distence_out = 0
        T_O_of_N_list = []
        for i in range(len(self.T_list)-1):
          T_O_of_N = np.dot(np.linalg.inv(np.dot(T_temp,self.eye2hand_T)),np.dot(self.T_list[i],self.eye2hand_T))
          T_O_of_N_inv = T_O_of_N
          # print("self.T_last",self.T_last)
          # print("T",self.T)
          # print("T_O_of_N",T_O_of_N)
          T_O_of_N = np.linalg.inv(T_O_of_N)
          T_O_of_N = np.delete(T_O_of_N,3,axis=0)
          T_O_of_N_list.append(np.delete(T_O_of_N_inv,3,axis=0))
          # print("T_O_of_N",T_O_of_N)
          K_T = np.dot(self.mtx,T_O_of_N)   #F
          # print("K_T",K_T)
          U_V_last = np.array([self.U_V_list[0][i],self.U_V_list[1][i],1])
          K_inv_U_V_last = np.dot(np.linalg.inv(self.mtx),U_V_last)
          # print("K_inv_U_V_last",K_inv_U_V_last)
          W1 = K_T[0][0]*K_inv_U_V_last[0]+K_T[0][1]*K_inv_U_V_last[1]+K_T[0][2]
          W2 = K_T[1][0]*K_inv_U_V_last[0]+K_T[1][1]*K_inv_U_V_last[1]+K_T[1][2]
          W3 = K_T[2][0]*K_inv_U_V_last[0]+K_T[2][1]*K_inv_U_V_last[1]+K_T[2][2]
          # W1_W3 = (np.dot(np.delete(T_O_of_N[0],-1,axis=1),K_inv_U_V_last.T))/(np.dot(np.delete(T_O_of_N[2],-1,axis=1),K_inv_U_V_last.T)+0.000000000000001)
          # print("W1",W1)
          # print("W2",W2)
          # print("W3",W3)
          Znu = abs(( K_T[0][3]-K_T[2][3]*W1/W3)/(Un-W1/W3))
          Znv = abs(( K_T[1][3]-K_T[2][3]*W2/W3)/(Vn-W2/W3))
        #   if Znu>3 and Znv<3:
        #     Znu = Znv
        #   if Znv>3 and Znu<3:
        #     Znv = Znu
          Kv = abs(Vn-self.U_V_list[1][i])
          Ku = abs(Un-self.U_V_list[0][i])
          if Kv<Ku:
            Kv = abs(Kv/Ku)**2
            Ku = 2-Kv
          else:
            Ku = abs(Ku/Kv)**2
            Kv = 2-Ku
          distence_ = (Znu*Ku+Znv*Kv)/2
          distence_all.append(distence_)
          dist_pose = 0
    
          for j in range(3):
            dist_pose += abs(self.T_list[i][j][3]-T_temp[j][3])
          k_dis.append(dist_pose)
          dist_pose_all += dist_pose

        for i in range(len(k_dis)):
          distence_out+= (k_dis[i]/dist_pose_all)*distence_all[i]
        print("===============================")
        print("distence_out",distence_out)
        print("distence_real",distence)
        self.distence_2.append(distence_out)
        self.distence_now.append(distence)
        self.time_list.append((time.time()-self.start_time))
        
        # U_V = np.array([Un,
        #                 Vn,
        #                 1])
        
        # last_cost = 0
        # distence_out_real = distence_out
        # erro_ = 0
        # for iter in range(10):
        #   erro_ = 0
        #   J = 1
        #   print('-----------------')
        #   Pn =  np.dot(np.linalg.inv(self.mtx),U_V)*distence_out
        #   # print("Pn",Pn)
        #   Pn_4 = np.array([Pn[0],
        #                   Pn[1],
        #                   Pn[2],
        #                   1])
        #   for i in range(len(self.T_list)-1):
        #                      # 平移矩阵
        #       # t_= np.array([T_O_of_N_list[i][0][3],T_O_of_N_list[i][1][3],T_O_of_N_list[i][2][3]])
        #       # R_i = np.array([T_O_of_N_list[i][0][0:3],T_O_of_N_list[i][1][0:3],T_O_of_N_list[i][2][0:3]])
        #       # R_total =  np.dot(self.mtx,(np.dot(R_i,np.dot(np.linalg.inv(self.mtx),U_V))*distence_out+t_))
        #       # a,b,c = np.dot(R_i,np.dot(np.linalg.inv(self.mtx),U_V))
        #       # R_total0 = (self.mtx[0][0]*a+self.mtx[0][2]*c)*distence_out +self.mtx[0][0]*t_[0]+self.mtx[0][2]*t_[2]
        #       # R_total1 = (self.mtx[1][1]*b+self.mtx[1][2]*c)*distence_out +self.mtx[1][1]*t_[1]+self.mtx[1][2]*t_[2]
        #       # R_total2 = c*distence_out +t_[2]
              
        #       P_i = np.dot(T_O_of_N_list[i],Pn_4) # 3D坐标点
        #       # print("P_i",P_i)
        #       # print("R_total",R_total/R_total[2])
        #       # print("R_total2",R_total0/R_total2,R_total1/R_total2)
        #       # print("real_Pi",self.P_list[i+1])
        #       New_u_v = np.dot(self.mtx,np.array(P_i).T)/(P_i[2])
        #       # print("New_u_v",New_u_v)
        #       # print("real_U_V",self.U_V_list[0][i],self.U_V_list[1][i])
        #       e = self.U_V_list[0][i]-New_u_v[0]+self.U_V_list[1][i]-New_u_v[1]
           
        #   #                     #
        #   #            # 深度距离的平方
        #       erro_ += abs(e)     # 向量的模 累
        #       # J += self.mtx[0][0]*(c**2*(t_[0]*c-t_[2]*a))/(c*t_[2]+c**2*distence_out) + self.mtx[1][1]*(c**2*(t_[1]*c-t_[2]*b))/(c*t_[2]+c**2*distence_out)
        #          # 以上建立雅克比矩阵
              

      
        #   if iter>0 and erro_>=last_cost:# 误差变大则退出
        #     J = J*(-1)
        #       # print('out!!!!!!!!')
        #       # print('cost = ', erro_,' last_cost = ',last_cost)
        #       # break
          
        #   print('cost = ', erro_,' last_cost = ',last_cost)
        #   print("J",J)
        #   last_cost = erro_
        #   x = erro_*J*(0.0005)
        #   print("x",x)
          
        #   distence_out += x                    # 更新位姿
        #   print("distence_out",distence_out)
        #   #print(np.linalg.norm(x))
   
       
        # self.distence_NL.append(distence_out)
        
        # print("Znn",Znu)
        # print("Znv",Znv)
        # print("distence",distence)


    
    def QR(self):
      while self.cap.isOpened():
        # prev_time = time.time()
        frame_QR = self.QRframe_queue.get()
        if frame_QR is not None:
          cv_image = frame_QR
          vel_msg = Twist()
          gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
          aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
          parameters =  aruco.DetectorParameters_create()
          #使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
          corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)
        #   bool_msg=Bool()
          if ids is not None:
                # print('---------------------')
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, self.mtx, self.dist)
                # 估计每个标记的姿态并返回值rvet和tvec ---不同
                (rvec-tvec).any() 
                for i in range(rvec.shape[0]):
                    aruco.drawAxis(cv_image, self.mtx, self.dist, rvec[i, :, :], tvec[i, :, :], 0.03)
                    aruco.drawDetectedMarkers(cv_image, corners)

                #将相机坐标系下目标位资投影到机械臂末端
                EulerAngles = self.rotationVectorToEulerAngles(rvec[0])
                pose_of_cam=np.array([tvec[0][0][0], tvec[0][0][1],tvec[0][0][2]])
                R_cam = self.ur_kmtic.eulerAnglesToRotationMatrix(EulerAngles)
                T_of_cam = np.array([[R_cam[0][0],R_cam[0][1],R_cam[0][2],pose_of_cam[0]],
                                [R_cam[1][0],R_cam[1][1],R_cam[1][2],pose_of_cam[1]],
                                [R_cam[2][0],R_cam[2][1],R_cam[2][2],pose_of_cam[2]],
                                [0,0,0,1]
                                ])
                T_pose_of_e = np.dot(self.eye2hand_T,T_of_cam)
                T = np.array(self.get_cur_T()).reshape(4,4)
                # T = np.array(self.get_cur_T())
                # R_traintion = np.array([T[0:3],T[4:7],T[8:11]])
                # Euler_traintion= Rotation.from_matrix(R_traintion).as_euler('zyx')
                # Euler_traintion=Euler_traintion*180/math.pi
               
                # print(T[3],T[7],T[11],Euler_traintion)

                T_base = np.dot(T,T_pose_of_e)
                R_ek = np.array([T_base[0][0:3],T_base[1][0:3],T_base[2][0:3]])
                Euler_ek=Rotation.from_matrix(R_ek).as_euler('zyx')
                print('----------------')
                print("====",T_base[0][3],T_base[1][3],T_base[2][3],Euler_ek[0], Euler_ek[1],Euler_ek[2])
                print('----------------')
                #将相机坐标系下目标位资投影到机械臂末端
                pose_of_cam=np.array([tvec[0][0][0], tvec[0][0][1],tvec[0][0][2]])
                pose_of_e = np.dot(self.eye2hand,pose_of_cam)
                EulerAngles = self.rotationVectorToEulerAngles(rvec[0])
                EulerAngles_180 = np.array(EulerAngles)*180/3.1415926
                
                # print(tvec[0][0][0], tvec[0][0][1],tvec[0][0][2],EulerAngles_180)
                # print('----------------')
                vel_msg.linear.x = (pose_of_e[0]- 0.2)#*0.5#*0.2
                vel_msg.linear.y = pose_of_e[1]+0.02188592
                vel_msg.linear.z = pose_of_e[2]+0.02901908
                # print('vel_msg.linear',vel_msg.linear.x,vel_msg.linear.y,vel_msg.linear.z)            
           
              
                k=-1
                if EulerAngles[0]>0:
                    k=1     
                vel_msg.angular.x = (EulerAngles[2])*0.3
                vel_msg.angular.y = (-EulerAngles[1])*0.3
                vel_msg.angular.z = ((EulerAngles[0]-k*3.1415926535898))*0.3
               
                if abs(vel_msg.angular.x) < 0.02:
                    vel_msg.angular.x = 0
                if abs(vel_msg.angular.y) < 0.02:
                    vel_msg.angular.y = 0
                if abs(vel_msg.angular.z) < 0.02:
                    vel_msg.angular.z = 0
                # print('vel_msg.angular',vel_msg.angular.x,vel_msg.angular.y,vel_msg.angular.z)

 
                self.first_distence = vel_msg.linear.x
                self.twist_pub_first.publish(vel_msg)
                

                # cv2.putText(cv_image, "Attitude_angle:" + str(EulerAngles), (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                #             cv2.LINE_AA)
                # cv2.putText(cv_image, "Posion:" + str(tvec), (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                #             cv2.LINE_AA)
        #   else:
        #       ##### DRAW "NO IDS" #####
      
        #       cv2.putText(cv_image, "No Ids", (0,64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
            #   print("No Ids")
        self.frame_QR_after = cv_image
        frame_resize = cv2.resize(self.frame_QR_after,(0,0),fx = 0.25,fy = 0.25,interpolation = cv2.INTER_NEAREST)
        cv2.imshow('frame', frame_resize)
        key = cv2.waitKey(10)
        if key == ord('q'):         # 按esc键退出
            print('show end')
            self.threading_flag = 0
            break
        if key == ord('s'):         # 按esc键退出
            self.hand.append(T[3])
            self.hand.append(T[7])
            self.hand.append(T[11])
            self.hand.append(Euler_traintion[0])
            self.hand.append(Euler_traintion[1])
            self.hand.append(Euler_traintion[2])
            self.camera.append(pose_of_cam[0])
            self.camera.append(pose_of_cam[1])
            self.camera.append(pose_of_cam[2])
            self.camera.append(EulerAngles_180[0])
            self.camera.append(EulerAngles_180[1])
            self.camera.append(EulerAngles_180[2])
        if key == ord('c'):
            eye_in_hand.e_h_H(self.hand,self.camera)
        # print("FPS_QR: {}".format(int(1/(time.time() - prev_time))))
        if not self.threading_flag:
              print('QR end')
              break
          # 显示结果框架
          # cv2.imshow("frameQR",cv_image)
          # fps = fps_queue.get()
          # key = cv2.waitKey(6)
          # if key == ord('q'):         # 按esc键退出
          #     print('esc break...')
          #     #rospy.signal_shutdown("c")
          #     break
          #     cv2.destroyAllWindows()
      #cv2.destroyAllWindows()
    
    def show(self):
        while self.threading_flag:
            
            # if self.frame_QR_after is not None:
            #   cv2.imshow("frame_QR",self.frame_QR_after)
            # if self.frame_yolo is not None:  
            #   cv2.imshow("frame_yolo",self.frame_yolo)
            
            # frameUp = np.hstack((self.frame_QR_after, self.frame_yolo))
            # frame_resize = cv2.resize(self.frame_yolo,(0,0),fx = 0.25,fy = 0.25,interpolation = cv2.INTER_NEAREST)
            frame_resize = cv2.resize(self.frame_QR_after,(0,0),fx = 0.25,fy = 0.25,interpolation = cv2.INTER_NEAREST)
            cv2.imshow('frame', frame_resize)
            key = cv2.waitKey(10)
            if key == ord('q'):         # 按esc键退出
                print('show end')
                self.threading_flag = 0
                break
            if(abs(self.first_distence)<0.0003) and (self.find_needle):
                            if not self.last_print_state_ttsp:
                                self.start_T = self.get_cur_T()
                                self.last_print_state_ttsp = True
                                self.last_print_state_btfp = False
                                print('========== turn to second part ==========')
    
                            bool_msg = False
                            self.part_pub.publish(bool_msg)
            if (not self.find_needle):
                    
                self.needle_lost_times += 1
                if self.needle_lost_times>50:
                    bool_msg = True
                    self.part_pub.publish(bool_msg)
                    if not self.last_print_state_btfp:
                        self.last_print_state_ttsp = False
                        self.last_print_state_btfp = True
                        print('========== back to first part ==========')
                        
        
        time.sleep(4)
        self.cap.release()
    #     plt.figure(0)
    # #   plt.plot(self.time_list, self.distence_u,label='U',color='g')
    # #   plt.plot(self.time_list, self.distence_now ,label='Real',color='r')
    # #   plt.plot(self.time_list, self.distence_v ,label='V',color='y')
    # #   plt.ylabel('Distance (m)')
    # #   plt.xlabel('time (s)')
    # #   plt.legend(loc='upper right')
    # #   plt.legend()
    # #   y_min,y_max =plt.ylim()

    # #   plt.figure(1)
    #     plt.plot(self.time_list, self.distence_2,label='MRE',color='b')
    #     plt.plot(self.time_list, self.distence_now ,label='Real',color='r')
    #     plt.ylabel('Distance (s)')
    #     plt.xlabel('time (s)')
    #     plt.legend(loc='upper right')
    # #   plt.ylim(y_min,y_max)
    #     plt.legend()
    #     plt.show()
    #     print("erros",np.sum(abs(np.array(self.distence_now)-np.array(self.distence_2)))/len(self.distence_2) )
        

    def rotationVectorToEulerAngles(self,rvec):
      R = np.zeros((3, 3), dtype=np.float64)
      cv2.Rodrigues(rvec, R)
      sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
      singular = sy < 1e-6
      if not singular:  # 偏航，俯仰，滚动
          x = math.atan2(R[2, 1], R[2, 2])
          y = math.atan2(-R[2, 0], sy)
          z = math.atan2(R[1, 0], R[0, 0])
      else:
          x = math.atan2(-R[1, 2], R[1, 1])
          y = math.atan2(-R[2, 0], sy)
          z = 0
      # 偏航，俯仰，滚动换成角度

      return x,y,z
    def dark_Thread(self):
        thresh = 0.5
        Thread(target=self.get_frame).start()
        # Thread(target=self.video_capture, args=(self.frame_queue,self.darknet_image_queue)).start()
        # Thread(target=self.inference, args=(self.darknet_image_queue, self.detections_queue, self.fps_queue,thresh)).start()
        # Thread(target=self.drawing, args=(self.frame_queue, self.detections_queue, self.fps_queue)).start()
        Thread(target=self.QR).start()
        # Thread(target=self.show).start()
      
    
     
if __name__ == "__main__":
    #rate = rospy.Rate(30) 
    Image_converter()

    
