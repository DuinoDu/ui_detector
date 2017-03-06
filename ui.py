#!/usr/bin/env python

'''
Add to tools/demo.py

def init():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    demo_net = 'vgg16' # zf
    modeltxt = os.path.join(cfg.MODELS_DIR, NETS[demo_net][0], 'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    modelbin = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models', NETS[demo_net][1])
    if not os.path.isfile(modelbin):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(modelbin))
    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0

    net = caffe.Net(modeltxt, modelbin, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(modeltxt)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    return net

def detect(net, imgPath):
    im = cv2.imread(imgPath)
    # Detect
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
    
        thresh = 0.5
        inds = np.where(dets[:, -1] >= thresh)[0]
        for i in inds:
            bbox = dets[i, :4]
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 5)
    return im

'''

# This is only needed for Python v2 but is harmless for Python v3.
import sip
sip.setapi('QString', 2)

from PyQt4 import QtCore, QtGui
import sys
sys.path.append("../tools")
import demo
import time
class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        """ Constructor initializes a default value for the brightness, creates
            the main menu entries, and constructs a central widget that contains
            enough space for images to be displayed.
        """
        super(MainWindow, self).__init__()
        self.net = demo.init()

        self.scaledImage = QtGui.QImage()
        self.path = ''
        self.fileName = []
        self.directoryFile = ''
        self.labelSize = QtCore.QSize(256,256)
        self.width = 600
        self.height = 800
        self.resize(800, 600)

        self.setWindowTitle("Detection")

        self.textEdit = QtGui.QTextEdit()
        self.setCentralWidget(self.textEdit)

        self.imageLabel = QtGui.QLabel()
        self.imageLabel.setFrameShadow(QtGui.QFrame.Sunken)
        self.imageLabel.setFrameShape(QtGui.QFrame.StyledPanel)
        self.imageLabel.setMinimumSize(self.labelSize)
        
        self.openButton = QtGui.QPushButton("Open")
        self.openButton.setMinimumSize(64, 32)
        #self.openButton.setMaximumSize(100,32)
        
        self.detectButton = QtGui.QPushButton("Detect")
        self.detectButton.setMinimumSize(64,32)

        #self.saveButton = QtGui.QPushButton("Save")
        #self.saveButton.setMinimumSize(32, 32)

        self.quitButton = QtGui.QPushButton("Quit")
        self.quitButton.setMinimumSize(64, 32)

        self.openButton.clicked.connect(self.chooseFile)
        self.detectButton.clicked.connect(self.detect)
        #self.saveButton.clicked.connect(self.saveImage)
        self.quitButton.clicked.connect(self.close)

        frame = QtGui.QFrame(self)
        grid = QtGui.QGridLayout(frame)
        grid.setSpacing(10)

        
        grid.addWidget(self.textEdit, 0,0,1,0)
        grid.addWidget(self.imageLabel, 0, 1, 1,2)
        #grid.addWidget(self.imageLabel, 0, 0)
        grid.addWidget(self.openButton, 2, 0)
        grid.addWidget(self.detectButton,2, 1)
       # grid.addWidget(self.saveButton, 2, 0)
        grid.addWidget(self.quitButton,2,2) 
        self.setCentralWidget(frame)

    def chooseFile(self):
        """ Provides a dialog window to allow the user to specify an image file.
            If a file is selected, the appropriate function is called to process
            and display it.
        """

        #imageFile = QtGui.QFileDialog.getOpenFileName(self,
        #      "Choose an image file to open", self.path, "Images (*.*)")
        
        #print(imageFile)
        directoryFile = QtGui.QFileDialog.getExistingDirectory()
        #print(directoryFile)
        
        import os
        FileName = []
        FileName = os.listdir(directoryFile)
        self.fileName = FileName
        self.directoryFile = directoryFile
        filename = []
        for name in FileName:
            filename.append(name)
            filename.append('\n')
        str = ('').join(filename)
        self.textEdit.setText(str)
        print(str)
            
    def openImageFile(self, imageFile):
        originalImage = QtGui.QImage()
        if originalImage.load(imageFile):
            self.setWindowTitle(imageFile)
            self.scaledImage = originalImage.scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio)
            self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(self.scaledImage))
        else:
            QtGui.QMessageBox.warning(self, "Cannot open file",
                    "The selected file could not be opened.",
                    QtGui.QMessageBox.Cancel, QtGui.QMessageBox.NoButton,
                    QtGui.QMessageBox.NoButton)

    def detect(self):
        import os
        for name in self.fileName:
            imageFile = os.path.join(self.directoryFile,name)
            self.path = imageFile
            result = demo.detect(self.net, self.path)
        #bgr = result
        #bgr[:,:,1] = result[:,:,0]
        #bgr[:,:,2] = result[:,:,1]
        #bgr[:,:,0] = result[:,:,2]
        #result = bgr
            height, width, channel = result.shape
            bytesPerLine = 3 * width
            resultImg = QtGui.QImage(result.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            self.scaledImage = resultImg.scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio)
            self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(self.scaledImage))
            time.sleep(3)

    def saveImage(self,imagename):
        """ Provides a dialog window to allow the user to save the image file.
        """
        #imageFile = QtGui.QFileDialog.getSaveFileName(self,
        #        "Choose a filename to save the image", "", "Images (*.png)")
        #print(imageFile)
        #info = QtCore.QFileInfo(imageFile)
        info = imagename
        #if info.baseName() != '':
        if info !='':
           # print(info)
           # newImageFile = QtCore.QFileInfo(info.absoluteDir(),
            #        info+'detected' + '.png').absoluteFilePath()
            newImageFile = '/home/fangfang/py-faster-rcnn/insulator/'+info
            print(newImageFile)
            if not self.imageLabel.pixmap().save(newImageFile, 'PNG'):
                QtGui.QMessageBox.warning(self, "Cannot save file",
                        "The file could not be saved.",
                        QtGui.QMessageBox.Cancel, QtGui.QMessageBox.NoButton,
                        QtGui.QMessageBox.NoButton)
        else:
            QtGui.QMessageBox.warning(self, "Cannot save file",
                    "Please enter a valid filename.", QtGui.QMessageBox.Cancel,
                    QtGui.QMessageBox.NoButton, QtGui.QMessageBox.NoButton)


if __name__ == '__main__':

    import sys
    app = QtGui.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
