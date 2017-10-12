import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib.image as im
#%matplotlib inline

def make_data(lineMu, lineSig, lineL, sqMu, sqSig, sqL, sqLocMu, sqLocSig, 
              recMu, recSig, recHor, recVer, recLocMu, recLocSig):
    """makes 512 x 512 images with normally distributed # of lines, squares, and rectangles
    of given sizes at uniformly distributed locations"""
    
    # randomly generate number of lines according to N(lineMu, lineSig)
    lineN = int(npr.normal(lineMu, lineSig))
    # number of lines has to be nonnegative
    if lineN < 0:
        lineN = 0
    # uniform-randomly generate position of line centers
    # note the 50-pixel buffer on the image edges
    # so that our shapes (lines, squares, rectangles) don't fall off
    # the image boundaries, i.e. the x and y coordinates are within (50, 462)
    linePosX = map(int,npr.uniform(size=lineN)*412+50)
    linePosY = map(int,npr.uniform(size=lineN)*412+50)
    linePos = zip(linePosX, linePosY)
    
    # same for squares
    sqN = int(npr.normal(sqMu, sqSig))
    if sqN < 0:
        sqN = 0
    sqPosX = map(int,npr.uniform(size=sqN)*412+50)
    sqPosY = map(int,npr.uniform(size=sqN)*412+50)
    sqPos = zip(sqPosX, sqPosY)
    
    # same for rectangles
    recN = int(npr.normal(recMu, recSig))
    if lineN < 0:
        lineN = 0
    recPosX = map(int,npr.uniform(size=recN)*412+50)
    recPosY = map(int,npr.uniform(size=recN)*412+50)
    recPos = zip(recPosX, recPosY)
    
    # initialize canvas
    fig = np.zeros((512,512))
    
    print "lineN: %d, sqN: %d, recN: %d" %(lineN, sqN, recN)
    print "linePos: ", linePos
    print "sqPos: ", sqPos
    print "recPos: ", recPos
    
    # draw left-diagonal lines
    for line_i in range(lineN):
        fig[ [range(int(linePos[line_i][1]-0.5*lineL/2**0.5), int(linePos[line_i][1]+0.5*lineL/2**0.5))  ],
             [range(int(linePos[line_i][0]-0.5*lineL/2**0.5), int(linePos[line_i][0]+0.5*lineL/2**0.5))  ] ] = 1
    
    for sq_i in range(sqN):
        fig[int(sqPos[sq_i][1]-0.5*sqL):int(sqPos[sq_i][1]+0.5*sqL), 
            int(sqPos[sq_i][0]-0.5*sqL):int(sqPos[sq_i][0]+0.5*sqL)] = 1
    
    for rec_i in range(recN):
        fig[int(recPos[rec_i][1]-0.5*recVer):int(recPos[rec_i][1]+0.5*recVer), 
            int(recPos[rec_i][0]-0.5*recHor):int(recPos[rec_i][0]+0.5*recHor)] = 1
    
    return fig

if __name__ == '__main__':
    fig = make_data(10, 1, 100, 3, 1, 40, 5, 1 ,3, 1, 40, 20, 5, 1)
    #plt.imshow(fig)
    im.imsave('imdata.png', fig)
