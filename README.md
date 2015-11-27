# Somnolence-Detector
#This is the bigining of the idea of how to make an open-source somnolence detector. Any help will be appreciated
#The main target is that it could be used embebed in a raspberry pi 2, but there are a lot of computational limitations yet. 
#This programa has been wrote in Python 2.7 and it makes use of Libraries as OpenCV.
#The last commented lines are from the uses of a SEN10171P digital light sensor.
#As code is presented here is posible to run it in a Laptop with Windows 8 as well as in raspbian.
#¿how to extract face landmarks once detected the region of interest of the face?

import numpy as np
import cv2
import video
##import smbus
import math
from common import draw_str, RectSelector

def rnd_warp(a):
    h, w = a.shape[:2]
    T = np.zeros((2, 3))
    coef = 0.2
    ang = (np.random.rand()-0.5)*coef
    c, s = np.cos(ang), np.sin(ang)
    T[:2, :2] = [[c,-s], [s, c]]
    T[:2, :2] += (np.random.rand(2, 2) - 0.5)*coef
    c = (w/2, h/2)
    T[:,2] = c - np.dot(T[:2, :2], c)
    return cv2.warpAffine(a, T, (w, h), borderMode = cv2.BORDER_REFLECT)

def divSpec(A, B):
    Ar, Ai = A[...,0], A[...,1]
    Br, Bi = B[...,0], B[...,1]
    C = (Ar+1j*Ai)/(Br+1j*Bi)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C

eps = 1e-5

        

class MOSSE():
    def __init__(self, frame, rect):
        x1, y1, x2, y2 = rect
        w, h = map(cv2.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        self.size = w, h
        img = cv2.getRectSubPix(frame, (w, h), (x, y))
    
        self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        g = np.zeros((h, w), np.float32)
        g[h//2, w//2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()

        self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)
        for i in xrange(3):
            a = self.preprocess(rnd_warp(img))
            A = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)
            
            self.H1 += cv2.mulSpectrums(self.G, A, 0, conjB=True)
            self.H2 += cv2.mulSpectrums(     A, A, 0, conjB=True)
         
        self.update_kernel()
        self.update(frame)

    def update(self, frame, rate = 0.125):
        global psr_
        (x, y), (w, h) = self.pos, self.size
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), (x, y))
        img = self.preprocess(img)
        self.last_resp, (dx, dy), self.psr = self.correlate(img)
        psr_=self.psr
        self.good = self.psr > 8.0
        if not self.good:
            return
        self.pos = x+dx, y+dy
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), self.pos)
        img = self.preprocess(img)
##        A = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
##        H1 = cv2.mulSpectrums(self.G, A, 0, conjB=True)
##        H2 = cv2.mulSpectrums(     A, A, 0, conjB=True)
##        self.H1 = self.H1 * (1.0-rate) + H1 * rate
##        self.H2 = self.H2 * (1.0-rate) + H2 * rate
##        self.update_kernel()

    @property
    def state_vis(self):
        f = cv2.idft(self.H, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
        h, w = f.shape
        f = np.roll(f, -h//2, 0)
        f = np.roll(f, -w//2, 1)
        kernel = np.uint8( (f-f.min()) / f.ptp()*255 )
        resp = self.last_resp
        resp = np.uint8(np.clip(resp/resp.max(), 0, 1)*255)

##        write(0x29,0x03)
##        value=(readWORD(address,0xAC))
##        print value
##        if value < 38:
##            diff=38-value
##            value=value+diff
##        if value > 39:
##            diff=value-39
##            value=value-diff
        #print value
        ret,crop_frameB = cv2.threshold(self.last_img,15,255,cv2.THRESH_BINARY)
        crop_frameB=smallmask1 & crop_frameB
        ret,crop_frameD = cv2.threshold(self.last_img,15,255,cv2.THRESH_BINARY)
        crop_frameD=smallmask & crop_frameD
        
        cv2.imshow("ojos",crop_frameD)
        cv2.imshow("boca",crop_frameB)
##        if circles is not None:
##                for i in circles[0,:]:
##                    cv2.circle(self.last_img,(i[0],i[1]),i[2],(0,0,255),2)
##                    cv2.circle(self.last_img,(i[0],i[1]),2,(255,0,0),3)
##                    print (i[0],i[1],i[2])
        vis = np.hstack([self.last_img, kernel, resp])
        self.last_img = cv2.cvtColor(self.last_img, cv2.COLOR_GRAY2BGR)
##        sum=crop_frameB.sum()
##        print  sum
        
        return vis

    def draw_state(self, vis):   
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))
        
        
        if self.good:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
        else:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.line(vis, (x2, y1), (x1, y2), (0, 0, 255))
        draw_str(vis, (x1, y2+16), 'PSR: %.2f' % self.psr)
        
    def preprocess(self, img):
        img = np.log(np.float32(img)+1.0)
        img = (img-img.mean()) / (img.std()+eps)
        return img*self.win

    def correlate(self, img):
        C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        h, w = resp.shape
        _, mval, _, (mx, my) = cv2.minMaxLoc(resp)
        side_resp = resp.copy()
        cv2.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval-smean) / (sstd+eps)
        return resp, (mx-w//2, my-h//2), psr

    def update_kernel(self):
        self.H = divSpec(self.H1, self.H2)
        self.H[...,1] *= -1

class App :
    def __init__(self, video_src):
        steady=1
        while True:
            ret, frame = video_capture.read() 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            heigth,width=gray.shape
            print heigth,width
            faces = faceCascade.detectMultiScale(gray, scaleFactor=3.2,minNeighbors=4, minSize=(150, 160), flags = cv2.CASCADE_SCALE_IMAGE)
            steady=steady+1
            if steady==11:
                steady=0
            if ( len(faces)>0) & (steady==10):
                break
            print "Not face detected"
        for (x, y, w, h) in faces:
            print 'Localizado'
##            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
##            cv2.putText(frame,'w: ' +str(heigth)+ ' h: '+str(width),(5,20),cv2.FONT_HERSHEY_SIMPLEX, .5,(0,0,0),2)
##            cv2.imshow('LocalizaciÃ³n de rostro', frame)
        _, self.frame = video_capture.read()
        self.trackers = []
##        X=50
##        Y=80
##        W=40
##        H=50
##        self.onrect([x+X,y+Y,x+W+X,y+H+Y])
##
##        X=140
##        Y=80
##        W=40
##        H=50
##        self.onrect([x+X,y+Y,x+W+X,y+H+Y])
        
        Xm=80
        Ym=160
        Wm=90
        Hm=55
        self.onrect([x+Xm,y+Ym,x+Wm+Xm,y+Hm+Ym])
        self.onrect([x,y,w+x,h+y])
       
      
        

    def onrect(self, rect):
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        tracker = MOSSE(frame_gray, rect)
        self.trackers.append(tracker)
        

    def run(self):
        while True:
            global psr_
            ret, self.frame = video_capture.read()
            frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            for tracker in self.trackers:
                    tracker.update(frame_gray)
            vis = self.frame.copy()
            for tracker in self.trackers:
                tracker.draw_state(vis)
            if len(self.trackers) > 0:
                cv2.imshow('tracker state', self.trackers[-1].state_vis)
            cv2.imshow('frame', vis)
            ch = cv2.waitKey(1) & 0xFF
            if ch == 27:
                break
            if psr_<=8:
                break

##bus = smbus.SMBus(1)
##address = 0x29
##
##
##cv2.imshow('Mask',smallmask)
##cv2.imshow('Mask1',smallmask1)
##
##
##def write(ADD,value):
##        bus.write_byte(ADD,value)
##        return -1
##
##def readWORD(ADD,value):
##        word=bus.read_word_data(ADD,value)
##        return word
##
##def readsen():
##        light = bus.read_byte(address)
##        return light
##cv2.waitKey(100)
##print 'Conectado sensor de Luz......'
##cv2.waitKey(100)
##print'..............'
##value=readsen()
##if value != 3:
##
##    print (value)
##    print ('Localizando')
##    cv2.waitKey(100)
##print'..............'
##cv2.waitKey(100)
##print'......'
##
##while value == 3:
##    write(0x29,0x03)
##    value=readWORD(address,0xAC)
##    print (value)
##    print ('Hecho')
##print (value)
##cv2.waitKey(100)
##print'Ok'
##cv2.waitKey(100)
##

mask=cv2.imread('C:/Users/Carlos Luna/Desktop/tablero_sencillo_blanco_y_negro.jpg')      
smallmask = cv2.resize(mask, (250,250))
smallmask=cv2.cvtColor(smallmask,cv2.COLOR_BGR2GRAY)
smallmask = cv2.GaussianBlur(smallmask,(5,5),3)

mask1=cv2.imread('C:/Users/Carlos Luna/Desktop/tablero_sencillo_blanco_y_negro.jpg')
smallmask1= cv2.resize(mask1, (250,250))
smallmask1=cv2.cvtColor(smallmask1,cv2.COLOR_BGR2GRAY)
smallmask1 = cv2.GaussianBlur(smallmask1,(5,5),3)

faceCascade = cv2.CascadeClassifier('../../data/haarcascades/haarcascade_frontalface_default.xml')
video_capture=cv2.VideoCapture(0)


while (1) :
        App(0).run()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
video_capture.release()
cv2.destroyAllWindows()

        
