
import colorsys
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import (QImage)
from numpy import ndarray

def datei_to_pixmap(datei_name):
    pixmap = QPixmap(datei_name)
    return pixmap
def arrayToQImage(array):
    from PyQt5.QtGui import (QImage)
    bytesperline = 3 * array.shape[1]
    if array.ndim == 2:
        from numpy import empty, uint8
        # expand to 3 channels 
        tmp = empty((array.shape[0],array.shape[1],3),dtype=uint8)
        tmp[:,:,0] = array
        tmp[:,:,1] = array
        tmp[:,:,2] = array
        
        array = tmp  
        return array
    # image creation   
    qImg = QImage(array.data, 
                  array.shape[1],
                  array.shape[0], 
                  bytesperline, QImage.Format_RGB888)
    return qImg
 
def array_to_pixmap(array):
    from PyQt5.QtGui import QPixmap
    return QPixmap(arrayToQImage(array))

def qImageToArray(qImg, grayscale=False, remove_alpha=True):
    from PyQt5.QtGui import (QImage)
    from numpy import ndarray, empty, uint8
    
    assert isinstance(qImg,QImage),'no QImage object'
    #qImg.convertToFormat(QImage.Format_BGR888) # added in Qt5.14
    qImg.convertToFormat(QImage.Format_RGB888)
    
    # create array from image data
    shape = [qImg.height(),qImg.width(),qImg.depth()//8]
    buffer = qImg.bits()
    buffer.setsize(shape[0]*shape[1]*shape[2])
    array = ndarray(shape=shape, buffer=buffer, dtype=uint8)
    
    if grayscale:
        out = empty((shape[0],shape[1]),dtype=uint8) 
        out = array[:,:,0]
    else:
        # create output without alpha channel
        if remove_alpha and shape[2] == 4:
            shape[2] = 3 
            out = empty(shape,dtype=uint8) 
        else:
            out = array
        # change from BGR to RGB
        out[:,:,2] = array[:,:,0]
        out[:,:,1] = array[:,:,1]
        out[:,:,0] = array[:,:,2]
        #out = array[:,:,-2:-5:-1] # not working
    return out



def pixmap_to_array(qPix):
    return qImageToArray(qPix.toImage())
