import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib.image as im
#%matplotlib inline

class toydata_gen(object):
    
    def __init__(self):
        self.image_height = 512
        self.image_width  = 512

        self.line_num_mean = 5
        self.line_num_std  = 3
        self.line_length   = 100

        self.square_num_mean = 1
        self.square_num_std  = 2
        self.square_length   = 20
        
        self.rect_num_mean = 1
        self.rect_num_std  = 2
        self.rect_length0  = 20
        self.rect_length1  = 40

    def num_classes(self):
        return 3

    def set_line(self,mean,std,length):
        (self.line_num_mean,self.line_num_std,self.line_length) = (mean,std,length)

    def set_square(self,mean,std,length):
        (self.square_num_mean,self.square_num_std,self.square_length) = (mean,std,length)

    def set_rect(self,mean,std,width,height):
        (self.rect_num_mean,self.rect_num_std,self.rect_length0,self.rect_length1) = (mean,std,width,height)

    def make_data(self,debug=False):
        """makes 512 x 512 images with normally distributed # of lines, squares, and rectangles
        of given sizes at uniformly distributed locations"""

        lineN = abs(int(npr.normal(self.line_num_mean, self.line_num_std)))
        # uniform-randomly generate position of line centers
        # note the 50-pixel buffer on the image edges
        # so that our shapes (lines, squares, rectangles) don't fall off
        # the image boundaries, i.e. the x and y coordinates are within (50, 462)
        linePosX = map(int,npr.uniform(size=lineN)*(self.image_width  - self.line_length))
        linePosY = map(int,npr.uniform(size=lineN)*(self.image_height - self.line_length))
        linePos  = zip(linePosX, linePosY)
        
        recN, sqN = (0,0)
        while (recN+sqN) < 1:
            
            # same for squares
            sqN = abs(int(npr.normal(self.square_num_mean, self.square_num_std)))
            sqPosX = map(int,npr.uniform(size=sqN)*(self.image_width  - self.square_length))
            sqPosY = map(int,npr.uniform(size=sqN)*(self.image_height - self.square_length))
            sqPos = zip(sqPosX, sqPosY)
            
            # same for rectangles
            recN = abs(int(npr.normal(self.rect_num_mean, self.rect_num_std)))
            recPosX = map(int,npr.uniform(size=recN)*(self.image_width  - self.rect_length0))
            recPosY = map(int,npr.uniform(size=recN)*(self.image_height - self.rect_length1))
            recPos = zip(recPosX, recPosY)
            
        # initialize canvas
        fig = np.zeros((self.image_width,self.image_height))

        rois = []

        if debug:
            print "lineN: %d, sqN: %d, recN: %d" %(lineN, sqN, recN)
            print "linePos: ", linePos
            print "sqPos: ", sqPos
            print "recPos: ", recPos
        
        # draw left-diagonal lines
        for line_i in range(lineN):
            fig[ [range(int(linePos[line_i][1]), int(linePos[line_i][1]+self.line_length/2**0.5))  ],
                 [range(int(linePos[line_i][0]), int(linePos[line_i][0]+self.line_length/2**0.5))  ] ] = 1
            
        for sq_i in range(sqN):
            x1,y1 = (int(sqPos[sq_i][0]),int(sqPos[sq_i][1]))
            x2,y2 = (x1+self.square_length,y1+self.square_length)
            fig[y1:y2, x1:x2] = 1
            rois.append([1,x1,y1,x2,y2])

        for rec_i in range(recN):
            x1,y1 = (int(recPos[rec_i][0]),int(recPos[rec_i][1]))
            x2,y2 = (x1 + self.rect_length0, y1+self.rect_length1)
            fig[y1:y2,x1:x2] = 1
            rois.append([2,x1,y1,x2,y2])    

        return fig,rois

    def forward(self):
        blob = {}
        img, roi = self.make_data()
        img = img.reshape([1,512,512,1])
        img = np.concatenate([img,img,img],axis=3)
        blob['data'] = img
        blob['im_info'] = [1,self.image_height,self.image_width,1]
        blob['gt_boxes'] = roi
        return blob

if __name__ == '__main__':
    g = toydata_gen()
    fig,rois = g.make_data()
    #plt.imshow(fig)
    im.imsave('imdata.png', fig)
    print rois
