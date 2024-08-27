
import os
import sys


from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
import tkinter as tk
from tkinter import filedialog

from TES import Ui_Form
from TES_algorithm import TES

sensors = []
dingbiao = []
channel = []


class MyMainForm(QMainWindow, Ui_Form):

    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        # 打开文件夹
        self.pushButton.clicked.connect(self.search_path1)
        self.pushButton_2.clicked.connect(self.search_path2)
        self.pushButton_3.clicked.connect(self.search_path3)

        # 关闭
        self.pushButton_5.clicked.connect(self.close)

        # 运行
        self.pushButton_4.clicked.connect(self.Run)

        # 参数
        # self.radioButton.clicked.connect(self.sensors_Click)
        # self.radioButton_2.clicked.connect(self.sensors_Click)
        #
        # self.radioButton_3.clicked.connect(self.dingbiao_Click)
        # self.radioButton_4.clicked.connect(self.dingbiao_Click)

        self.checkBox.stateChanged.connect(self.channel_Click)
        self.checkBox_2.stateChanged.connect(self.channel_Click)
        self.checkBox_3.stateChanged.connect(self.channel_Click)
        self.checkBox_4.stateChanged.connect(self.channel_Click)

    def search_path1(self):
        # 实例化
        root = tk.Tk()
        root.withdraw()
        Folderpath= filedialog.askdirectory()
        if os.path.exists(Folderpath):
            # self.open_file(Folderpath)
            self.lineEdit.setText(Folderpath)
        # Filepath= filedialog.askopenfilename()
        print('\n获取的文件夹地址：', Folderpath)
    def search_path2(self):
        # 实例化
        root = tk.Tk()
        root.withdraw()
        Folderpath= filedialog.askdirectory()
        if os.path.exists(Folderpath):
            # self.open_file(Folderpath)
            self.lineEdit_2.setText(Folderpath)
        # Filepath= filedialog.askopenfilename()
        print('\n获取的文件夹地址：', Folderpath)
    def search_path3(self):
        # 实例化
        root = tk.Tk()
        root.withdraw()
        Folderpath= filedialog.askdirectory()
        if os.path.exists(Folderpath):
            # self.open_file(Folderpath)
            self.lineEdit_3.setText(Folderpath)
        # Filepath= filedialog.askopenfilename()
        print('\n获取的文件夹地址：', Folderpath)
    # def open_file(self, path):
    #     files = os.listdir(path)  # 得到文件夹下的所有文件名称
    #     file_names = []
    #     for file in files:  # 遍历文件夹
    #         if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
    #             file_name = path + "/" + file  # 打开文件
    #             file_names.append(file_name)
    #         print("获取的文件地址：", file_names)  # 打印结果

    def closeEvent(self, event):
        print("关闭操作")
        result = QMessageBox.question(self, '关闭窗口',  '是否确认关闭',
                                      QMessageBox.Yes | QMessageBox.No,
                                      QMessageBox.No)

        if result == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()



    def channel_Click(self):
            global channel
            channel.clear()
            if self.checkBox.isChecked():
                channel.append(self.checkBox.text())
            if self.checkBox_2.isChecked():
                channel.append(self.checkBox_2.text())
            if self.checkBox_3.isChecked():
                channel.append(self.checkBox_3.text())
            if self.checkBox_4.isChecked():
                channel.append(self.checkBox_4.text())

            print(channel)

    def Run(self):
        if self.radioButton.isChecked():
            sensor = "GF-5A"
        elif self.radioButton_2.isChecked():
            sensor = "GF-5B"
        else:
            QMessageBox.about(self, '错误警告', '请选择传感器')
            return


        if len(channel) < 3:
            QMessageBox.about(self, '错误警告', '请输入至少三个通道')
            return
        else:
            # 逻辑处理
            # 选择通道的通道响应函数
            print("使用所选用的通道")


        if self.radioButton_3.isChecked():
            coffient = "卫星xml文件"
        elif self.radioButton_4.isChecked():
            coffient = "自定义文件"
        else:
            QMessageBox.about(self, '错误警告', '请选择定标文件')
            return

        img_path = self.lineEdit.text()
        atm_path = self.lineEdit_2.text()
        output_path = self.lineEdit_3.text()
        if (atm_path == "卫星影像文件夹/文件名" or not atm_path or
                img_path == "卫星影像文件夹/文件名" or not img_path or
                output_path == "卫星影像文件夹/文件名" or not output_path):
            QMessageBox.about(self, '错误警告', '未选择文件夹路径')
            return

        TES(img_path, atm_path, output_path)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainForm()
    myWin.show()
    sys.exit(app.exec_())



    # def sensors_Click(self, checked):
    #     if checked:
    #         global sensors
    #         if self.sender() == self.radioButton:
    #             sensors.clear()
    #             sensors.append('GF-5A')
    #             print("选择了GF-5A")
    #             # 这里可以添加其他处理逻辑，如更新UI等
    #         elif self.sender() == self.radioButton_2:
    #             sensors.clear()
    #             sensors.append('GF-5B')
    #             print("选择了GF-5B")
    #             # 同样可以添加其他处理逻辑
    #     print(sensors)
    #
    # def dingbiao_Click(self, checked):
    #     if checked:
    #         global dingbiao
    #         if self.sender() == self.radioButton_3:
    #             dingbiao.clear()
    #             dingbiao.append('卫星xml文件')
    #             print("卫星xml文件")
    #             # 这里可以添加其他处理逻辑，如更新UI等
    #         elif self.sender() == self.radioButton_4:
    #             dingbiao.clear()
    #             dingbiao.append('自定义')
    #             print("自定义")
    #             # 同样可以添加其他处理逻辑
    #
    #     print(dingbiao)