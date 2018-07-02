from __future__ import division

import os
import sys
import cv2
import time
import datetime
import numpy as np
import tkFileDialog
import Tkinter as tk
import tkFont as tkfont
from PIL import Image, ImageTk

#######################################################################################
#######################################################################################
#######################################################################################

reload(sys)
sys.setdefaultencoding('utf8')
os.chdir(sys.argv[1])

sys.path.insert(0, "/home/emre/.local/install/caffe/python")

import caffe

caffe.set_mode_gpu()
caffe.set_device(0)

mean_file = "../models/5/mean.npy"
mean = np.load(mean_file)

result_5 = {"0": "CIRKIN", "1": "VASAT", "2": "ORTA", "3": "GUZEL", "4": "COK GUZEL"}
result_3 = {"0": "CIRKIN", "1": "ORTA", "2": "GUZEL"}


#######################################################################################
#######################################################################################
#######################################################################################


def image_resize(image, width=None, height=None, inter=cv2.INTER_CUBIC):

    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def compute_kappa(class_size, test_size, confusion_matrix):

    sum_of_rows = []
    sum_of_cols = []
    for i in range(int(class_size)):

        sum_row = 0
        sum_col = 0
        for j in range(int(class_size)):
            sum_row += confusion_matrix[i][j]
            sum_col += confusion_matrix[j][i]

        sum_of_rows.append(sum_row)
        sum_of_cols.append(sum_col)

    sum = 0
    accuracy = 0
    for i in range(class_size):
        sum += sum_of_rows[i] * sum_of_cols[i]
        accuracy += confusion_matrix[i][i]

    chanceAgreement = sum / (test_size ** 2)
    kappa = (accuracy / test_size - chanceAgreement) / (1 - chanceAgreement)

    return kappa, accuracy


def get_images_and_labels(val_file, test_size):

    test = open(val_file, "r")

    inputs = []
    labels = []
    image_list = []

    for i in range(test_size):
        token = test.readline()
        path = token.split()[0]
        label = token.split()[1]

        inputs.append(caffe.io.load_image(path))
        labels.append(label)
        image_list.append(path)

    test.close()

    return inputs, image_list, labels


def select_deploy(page):

    cwd = os.getcwd()

    filename = tkFileDialog.askopenfilename(initialdir= cwd + "/..", title="Select file",
                                            filetypes=(("all files", "*.*"), ("jpeg files", "*.jpg")))
    
    if len(filename) != 0:
        page.model_file = filename
        index = filename.rfind("/")
        print_model_file = filename[index + 1:]

        page.text.configure(state='normal')
        page.text.insert('end', "Model file: " + print_model_file + "\n")
        page.text.configure(state='disabled')


def select_caffemodel(page):

    cwd = os.getcwd()

    filename = tkFileDialog.askopenfilename(initialdir= cwd + "/..", title="Select file",
                                            filetypes=(("all files", "*.*"), ("jpeg files", "*.jpg")))
    
    if len(filename) != 0:
        page.pretrained_model = filename

        index = filename.rfind("/")
        print_pretrained_model = filename[index + 1:]

        information = filename[0:index]
        page.class_size = information[information.rfind("/") + 1:]
        page.text.configure(state='normal')
        page.text.insert('end', "Class size: " + page.class_size + "\n")
        page.text.insert('end', "Pretrained file: " + print_pretrained_model + "\n")
        page.text.configure(state='disabled')


def predict_image(page):

    if page.filename is None or len(page.filename) == 0:
        page.text.configure(state='normal')
        page.text.insert('end', "Lutfen resim yukleyiniz!!!\n")
        page.text.configure(state='disabled')
    else:
        start_time = time.time()

        image = cv2.imread(page.filename, cv2.IMREAD_COLOR)

        y, x, channel = image.shape

        if y >= x:
            resized_image = image_resize(image, height=256)
        else:
            resized_image = image_resize(image, width=256)

        new_image = np.zeros((256, 256, 3), dtype=np.uint8)

        y, x, channel = resized_image.shape

        if y >= x:
            a = int((256 - x) / 2)
            if x != y:
                new_image[:y, a:x + a, :channel] = resized_image
            else:
                new_image[:y, :x, :channel] = resized_image
        else:
            a = int((256 - y) / 2)
            new_image[a:y + a, :x, :channel] = resized_image

        cv2.imwrite("../db/images/temp.jpg", new_image)
        ###############################################

        inputs = [caffe.io.load_image(page.filename)]

        net = caffe.Classifier(str(page.model_file), str(page.pretrained_model), image_dims=(256, 256),
                               mean=mean, raw_scale=255, channel_swap=(2, 1, 0))
        prediction = net.predict(inputs)

        end_time = time.time()

        if str(page.class_size) == "5":
            result = result_5
        else:
            result = result_3

        page.text.configure(state='normal')
        page.text.insert('end', "----------------------------------- RESULT ------------------------------------\n")

        for i in range(0, int(page.class_size)):
            page.text.insert('end', str(format(prediction[0][i], "0.6f")) + "\t" + "'"  + result[str(i)] + "'\n")

        page.text.insert('end', "Result: " + page.filename + " ---> " + "'" + str(prediction[0].argmax()) + "'\n")
        page.text.insert('end', "Execution time: " + str(format(end_time - start_time, "0.3f") + " seconds\n"))
        page.text.insert('end', "-----------------------------------------------------------------------------------\n")
        page.text.configure(state='disabled')


#######################################################################################
#######################################################################################
#######################################################################################


class Application(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")

        # get screen width and height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        width = 900
        height = 800

        # calculate position x and y coordinates
        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)
        self.geometry('%dx%d+%d+%d' % (width, height, x, y))
        self.resizable(width=tk.FALSE, height=tk.FALSE)
        self.title("Attractiveness Measurement Tool")

        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others

        container = tk.Frame(self, bg="red")
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, CameraPage, PicturePage, TestPage):

            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''

        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        header = tk.Frame(self)
        header.pack(side="top")

        #############################################################################

        self.button_home_mode = tk.Button(header, text="Home", bg="black", fg="white",
                                          command=lambda: controller.show_frame("StartPage"))

        self.button_camera_mode = tk.Button(header, text="Camera Mode",bg="black", fg="white",
                                            command=lambda: controller.show_frame("CameraPage"))

        self.button_picture_mode = tk.Button(header, text="Picture Mode", bg="black", fg="white",
                                          command=lambda: controller.show_frame("PicturePage"))

        self.button_test_mode = tk.Button(header, text="Test Mode", bg="black", fg="white",
                                            command=lambda: controller.show_frame("TestPage"))

        self.button_home_mode.pack(side="left", padx=10, pady=10)
        self.button_camera_mode.pack(side="left", padx=10, pady=10)
        self.button_picture_mode.pack(side="left", padx=10, pady=10)
        self.button_test_mode.pack(side="left", padx=10, pady=10)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")

        information = tk.Label(self, text="Welcome!!! Are you ready for predicting attractiveness?\n\n" +
                               "Camera Mode  ---> Kameradan goruntu alip test etmenizi saglar.\n\n" +
                               "Picture Mode ---> Diskten bir goruntu secip test etmenizi saglar.\n\n" +
                               "Test Mode ---> Diskten bir klasor secip kappa ve accuracy hesaplar.")

        information.configure(font=("Helvetica", 16, "bold", "italic"))
        information.pack(pady=150)


class CameraPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        header = tk.Frame(self)
        header.pack(side="top")

        ######################################################################################
        # bu block header daki butonlari tutar.

        self.button_home_mode = tk.Button(header, text="Home", bg="black", fg="white",
                                          command=lambda: controller.show_frame("StartPage"))

        self.button_camera_mode = tk.Button(header, text="Camera Mode", bg="black", fg="white",
                                            command=lambda: controller.show_frame("CameraPage"))

        self.button_picture_mode = tk.Button(header, text="Picture Mode", bg="black", fg="white",
                                             command=lambda: controller.show_frame("PicturePage"))

        self.button_test_mode = tk.Button(header, text="Test Mode", bg="black", fg="white",
                                          command=lambda: controller.show_frame("TestPage"))

        self.button_home_mode.pack(side="left", padx=10, pady=10)
        self.button_camera_mode.pack(side="left", padx=10, pady=10)
        self.button_picture_mode.pack(side="left", padx=10, pady=10)
        self.button_test_mode.pack(side="left", padx=10, pady=10)

        #######################################################################################
        # camera panelini tutar.

        self.camera = tk.Frame(self, width=640, height=480)
        self.camera.pack(pady=25)

        self.panel = tk.Label(self.camera, bg="black")
        self.panel.pack()

        frame = cv2.imread("../db/icons/camera.jpg", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.panel.configure(image=image)
        self.panel.image = image

        #######################################################################################
        # camera altindaki butonlari tutar.

        self.button_frame = tk.Frame(self)
        self.button_frame.pack()

        self.deploy_btn = tk.Button(self.button_frame, text="1- Select deploy.prototxt!!!", command=lambda: select_deploy(self),
                            background="black", foreground="white")
        self.deploy_btn.pack(side="left")

        self.caffemodel_btn = tk.Button(self.button_frame, text="2- Select .caffemodel!!!", command=lambda: select_caffemodel(self),
                             background="black", foreground="white")
        self.caffemodel_btn.pack(side="left", padx=10)

        self.btn = tk.Button(self.button_frame, text="3- Take a Snapshot!!!", command=self.take_snapshot,
                             background="black", foreground="white")
        self.btn.pack(side="left")

        self.predict_button = tk.Button(self.button_frame, text="4- Predict an Image!!!", command=lambda: predict_image(self),
                                        background="black", foreground="white")
        self.predict_button.pack(side="left", padx=10)

        #######################################################################################
        # output text box
        # create a Frame for the Text and Scrollbar

        txt_frm = tk.Frame(self)
        txt_frm.pack(pady=10)
        txt_frm.grid_propagate(False)
        txt_frm.grid_rowconfigure(0, weight=1)
        txt_frm.grid_columnconfigure(0, weight=1)

        # create a Text widget
        self.text = tk.Text(txt_frm, borderwidth=3, relief="sunken", width=70, height=10, state='disabled')
        self.text.config(font=("consolas", 10), undo=True, wrap='word')
        self.text.pack()
        # self.txt.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # create a Scrollbar and associate it with txt
        scrollb = tk.Scrollbar(txt_frm, command=self.text.yview)
        scrollb.grid(row=0, column=1, sticky='nsew')
        self.text['yscrollcommand'] = scrollb.set

        self.text.configure(state='normal')
        self.text.insert('end', '[INFO] Starting Camera Mode\n')
        self.text.configure(state='disabled')
        ######################################################################################

        self.model_file = None
        self.pretrained_model = None
        self.class_size = None
        self.filename = None

    def take_snapshot(self):

        cap = cv2.VideoCapture(0)
        while (True):

            # Capture frame-by-frame
            ret, frame = cap.read()

            # Display the resulting frame
            cv2.imshow('frame', frame)

            # press the letter "q" to save the picture
            if cv2.waitKey(1) & 0xFF == ord('q'):

                # write the captured image with this name
                ts = datetime.datetime.now()
                self.filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
                self.text.configure(state='normal')
                self.text.insert('end', "[INFO] {}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S")) + str("\n"))
                self.text.configure(state='disabled')
                self.filename = "../db/images/" + self.filename
                cv2.imwrite(self.filename, frame)
                break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.panel.configure(image=image)
        self.panel.image = image

        cap.release()
        cv2.destroyAllWindows()


class PicturePage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        header = tk.Frame(self)
        header.pack(side="top")

        ###############################################

        self.button_home_mode = tk.Button(header, text="Home", bg="black", fg="white",
                                          command=lambda: controller.show_frame("StartPage"))

        self.button_camera_mode = tk.Button(header, text="Camera Mode", bg="black", fg="white",
                                            command=lambda: controller.show_frame("CameraPage"))

        self.button_picture_mode = tk.Button(header, text="Picture Mode", bg="black", fg="white",
                                             command=lambda: controller.show_frame("PicturePage"))

        self.button_test_mode = tk.Button(header, text="Test Mode", bg="black", fg="white",
                                          command=lambda: controller.show_frame("TestPage"))

        self.button_home_mode.pack(side="left", padx=10, pady=10)
        self.button_camera_mode.pack(side="left", padx=10, pady=10)
        self.button_picture_mode.pack(side="left", padx=10, pady=10)
        self.button_test_mode.pack(side="left", padx=10, pady=10)

        ################################################

        self.camera = tk.Frame(self, width=640, height=480)
        self.camera.pack(pady=25)

        self.panel = tk.Label(self.camera, bg="black")
        self.panel.pack()

        frame = cv2.imread("../db/icons/camera.jpg", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.panel.configure(image=image)
        self.panel.image = image

        ################################################

        self.button_frame = tk.Frame(self)
        self.button_frame.pack()

        self.deploy_btn = tk.Button(self.button_frame, text="1- Select deploy.prototxt!!!",
                                    command=lambda: select_deploy(self),
                                    background="black", foreground="white")
        self.deploy_btn.pack(side="left")

        self.caffemodel_btn = tk.Button(self.button_frame, text="2- Select .caffemodel!!!",
                                        command=lambda: select_caffemodel(self),
                                        background="black", foreground="white")
        self.caffemodel_btn.pack(side="left", padx=10)

        self.btn = tk.Button(self.button_frame, text="3- Load an image!!!", command=self.load_file,
                             background="black", foreground="white")
        self.btn.pack(side="left")

        self.predict_button = tk.Button(self.button_frame, text="4- Predict an Image!!!",
                                        command=lambda: predict_image(self),
                                        background="black", foreground="white")
        self.predict_button.pack(side="left", padx=10)

        ###################################################
        # create a Frame for the Text and Scrollbar

        txt_frm = tk.Frame(self)
        txt_frm.pack(pady=10)
        txt_frm.grid_propagate(False)
        txt_frm.grid_rowconfigure(0, weight=1)
        txt_frm.grid_columnconfigure(0, weight=1)

        # create a Text widget
        self.text = tk.Text(txt_frm, borderwidth=3, relief="sunken", width=70, height=10, state='disabled')
        self.text.config(font=("consolas", 10), undo=True, wrap='word')
        self.text.pack()
        # self.txt.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # create a Scrollbar and associate it with txt
        scrollb = tk.Scrollbar(txt_frm, command=self.text.yview)
        scrollb.grid(row=0, column=1, sticky='nsew')
        self.text['yscrollcommand'] = scrollb.set

        self.text.configure(state='normal')
        self.text.insert('end', '[INFO] Starting Picture Mode\n')
        self.text.configure(state='disabled')

        ####################################################

        self.model_file = None
        self.pretrained_model = None
        self.class_size = None
        self.filename = None

    def load_file(self):

        cwd = os.getcwd()

        index = cwd.rfind("Desktop")

        path = cwd[:index] + "Desktop"

        name = tkFileDialog.askopenfilename(initialdir=path, title="Select file",
                                     filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))

        if len(name) != 0:
            self.filename = name

        if self.filename is not None:
            # self.text.configure(state='normal')
            # self.text.insert('end', '[INFO] Loading image file\n')
            # self.text.configure(state='disabled')

            frame = cv2.imread(self.filename, cv2.IMREAD_COLOR)

            y, x, channel = frame.shape

            if y >= x:
                resized_image = image_resize(frame, height=256)
            else:
                resized_image = image_resize(frame, width=256)

            new_image = np.zeros((256, 256, 3), dtype=np.uint8)
            y, x, channel = resized_image.shape

            if y >= x:
                a = int((256 - x) / 2)
                if x != y:
                    new_image[:y, a:x + a, :channel] = resized_image
                else:
                    new_image[:y, :x, :channel] = resized_image
            else:
                a = int((256 - y) / 2)
                new_image[a:y + a, :x, :channel] = resized_image

            image = image_resize(new_image, height=480)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            self.panel.configure(image=image)
            self.panel.image = image

        print self.filename


class TestPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        header = tk.Frame(self)
        header.pack(side="top")

        self.button_home_mode = tk.Button(header, text="Home", bg="black", fg="white",
                                          command=lambda: controller.show_frame("StartPage"))

        self.button_camera_mode = tk.Button(header, text="Camera Mode", bg="black", fg="white",
                                            command=lambda: controller.show_frame("CameraPage"))

        self.button_picture_mode = tk.Button(header, text="Picture Mode", bg="black", fg="white",
                                             command=lambda: controller.show_frame("PicturePage"))

        self.button_test_mode = tk.Button(header, text="Test Mode", bg="black", fg="white",
                                          command=lambda: controller.show_frame("TestPage"))

        self.button_home_mode.pack(side="left", padx=10, pady=10)
        self.button_camera_mode.pack(side="left", padx=10, pady=10)
        self.button_picture_mode.pack(side="left", padx=10, pady=10)
        self.button_test_mode.pack(side="left", padx=10, pady=10)

        ##########################################################

        # create a Frame for the Text and Scrollbar
        txt_frm = tk.Frame(self)
        txt_frm.pack(side="top", pady=50)
        txt_frm.grid_propagate(False)
        txt_frm.grid_rowconfigure(0, weight=1)
        txt_frm.grid_columnconfigure(0, weight=1)

        # create a Text widget
        self.text = tk.Text(txt_frm, borderwidth=3, relief="sunken", width=70, height=10, state='disabled')
        self.text.config(font=("consolas", 10), undo=True, wrap='word')
        self.text.pack()
        # self.txt.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # create a Scrollbar and associate it with txt
        scrollb = tk.Scrollbar(txt_frm, command=self.text.yview)
        scrollb.grid(row=0, column=1, sticky='nsew')
        self.text['yscrollcommand'] = scrollb.set

        self.text.configure(state='normal')
        self.text.insert('end', '[INFO] Starting Test Mode\n')
        self.text.configure(state='disabled')

        #############################################################

        self.dir_files = None

        self.button_frame = tk.Frame(self)
        self.button_frame.pack()

        self.button_deploy = tk.Button(self.button_frame, text="1- Select deploy.prototxt!!!", command=lambda: select_deploy(self),
                                       background="black", foreground="white")
        self.button_deploy.pack(side="top", pady=10)

        self.button_caffemodel = tk.Button(self.button_frame, text="2- Select caffemodel!!!", command=lambda: select_caffemodel(self),
                                     background="black", foreground="white")
        self.button_caffemodel.pack(side="top", pady=10)

        self.button_predict = tk.Button(self.button_frame, text="3- Start test!!!", command=self.predict,
                                        background="black", foreground="white")
        self.button_predict.pack(side="top", pady=10)

        #############################################################

        self.model_file = "../models/5/deploy.prototxt"
        self.pretrained_model = "../models/5/alexnet_10000.caffemodel"
        self.class_size = 5


    def predict(self):

        test_size = 100

        # input array, image paths and labels alinir.
        inputs, image_list, labels = get_images_and_labels("../db/val_rndm.txt", 100)

        # confusion matrix hazirlanir.
        confusion_matrix = [[0 for x in range(int(self.class_size))] for y in range(int(self.class_size))]

        net = caffe.Classifier(str(self.model_file), str(self.pretrained_model), image_dims=(256, 256),
                               mean=mean, raw_scale=255, channel_swap=(2, 1, 0))
        prediction = net.predict(inputs)

        for i in range(test_size):
            result = prediction[i].argmax()
            confusion_matrix[int(labels[i])][int(result)] += 1
            # print image_list[i], "-", labels[i], "predicted ->", result

        kappa, accuracy = compute_kappa(int(self.class_size), test_size, confusion_matrix)
        self.text.configure(state='normal')
        self.text.insert('end', "----------------------------------- RESULT ------------------------------------\n")
        self.text.insert('end', "Kappa: " + str(kappa) + "\n")
        self.text.insert('end', "Accuracy: " + str(accuracy / 100) + "\n")
        self.text.insert('end', "-----------------------------------------------------------------------------------\n")
        self.text.configure(state='disabled')

#######################################################################################
#######################################################################################
#######################################################################################


if __name__ == '__main__':

    app = Application()
    app.mainloop()
    app.quit()