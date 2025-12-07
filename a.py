#  -*- coding: utf-8 -*-



from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap, QMovie
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication , QMainWindow, QHeaderView, QMessageBox, QWidget
from PyQt5.QtCore import pyqtSignal , Qt
from PyQt5.QtGui import QStandardItemModel,QStandardItem, QGuiApplication
from PyQt5 import QtCore
from PyQt5.QtWidgets import QProgressDialog

QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)


import numpy as np
import torch
import os
from torch.utils.data import DataLoader
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.cm as cm


from model.pointnet import PointNet
from utils import data_reading, get_testing_patches, pc_normalize_all
from dataset import MyDataset_test

from bag import xai_pointcloud
from bag.utils.generic_utils import segment_fn
from bag.utils.show_result import saliency_map, plot_result

from untitled import Ui_MainWindow
import warnings
warnings.filterwarnings('ignore')




class Stream(QObject):
    """Redirects console output to text widget."""
    newText = pyqtSignal(str)
    def write(self, text):
        self.newText.emit(str(text))



def matplotlib_gif(data, out_path, name, data_pattern):
    X = data[:, 0]
    Y = data[:, 1]

    x_range = X.max() - X.min()
    y_range = Y.max() - Y.min()

    fig, ax = plt.subplots(figsize=(x_range / 100, y_range / 100))
    ax.set_xlim((X.min()-10, X.max()+10))
    ax.set_ylim((Y.min()-10, Y.max()+10))

    # 只保留四个边框，移除坐标轴刻度及其标签
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
                   labelleft=False)

    # 添加网格
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    line, = ax.plot([], [], 'o-', lw=2, c='#00008B')

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        x0 = X[:frame]
        y0 = Y[:frame]
        line.set_data(x0, y0)
        return line,

    ani = animation.FuncAnimation(fig, update, frames=range(0, len(X), 32), init_func=init, blit=True)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    ani.save(os.path.join(out_path, f'{name}-{str(data_pattern)}.gif'), writer='pillow', fps=5)
    plt.close()


def matplotlib_result(loc_pre_s, out_path, name, data_pattern):
    # 数据
    labels = ['HC', 'PD']
    colors = ['#00008B', '#8B0000']  # 深蓝色和深红色
    explode = (0, 0.1)  # 凸显PD部分

    # 绘制饼状图
    fig, ax = plt.subplots(figsize=(8,8))
    wedges, texts, autotexts = ax.pie(loc_pre_s, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%', startangle=140,
           wedgeprops=dict(width=0.8), textprops={'fontsize': 30, 'fontweight': 'bold'})

    # 设置标签文本颜色为黑色
    for text in texts:
        text.set_color('black')

    # 设置百分比文本颜色为白色
    for autotext in autotexts:
        autotext.set_color('white')

    # 确保饼图是圆形
    ax.axis('equal')
    plt.savefig(os.path.join(out_path, f'{name}-{str(data_pattern)}.jpg'), dpi=800)
    plt.close()


def batch_predict(pc, window_size, stride_size, model_name, label_type, device, model):
    pred_labels = []  # For each sequence, store the true category and model prediction category of the segmented patc
    patch_dataset = get_testing_patches(pc, window_size, stride_size)
    test_dataset = MyDataset_test(dataset=patch_dataset, name=label_type, transform=None)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)
    # test_bar = tqdm(test_loader)
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        with torch.no_grad():
            if model_name == 'PointNet':
                pred, trans_feat = model(inputs)
            pred = torch.max(pred, dim=-1)[1]

            pred_data = pred.data.cpu().detach().numpy().flatten()
            for k in np.arange(len(inputs)):
                pred_labels.append(pred_data[k])

    result = len([value for value in pred_labels if value == 1]) / len(pred_labels) # PD类别patches占比
    prob = np.array([1-result, result], dtype=np.float32)
    return prob











class ExplanationWorker(QThread):
    finished = pyqtSignal()
    resultReady = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = None
        self.pattern = None
        self.model_name = None
        self.neighbor_num = None

    def set_parameters(self, name, pattern, model_name, neighbor_num):
        self.name = name
        self.pattern = pattern
        self.model_name = model_name
        self.neighbor_num = neighbor_num

    def run(self):
        try:
            metrics = self.explanation()  # 获取解释结果
            self.resultReady.emit(metrics)  # 发送解释结果信号
        except Exception as e:
            print(f"Error in ExplanationWorker: {e}")
        finally:
            self.finished.emit()

    def explaintion(self):
        # 根据你的代码进行修改，这里仅展示了示例
        name = self.name
        pattern = self.pattern
        model_name = self.model_name
        neighbor_num = self.neighbor_num

        dataset = 'ParkinsonHW'
        data_path = f'PD/{name}.txt'
        label_type = data_path.strip().split('/')[0]
        method = 'lime'
        column_name = ['x', 'y', 'r']
        influ_name = [2]  # 查看哪些特征对模型结果的影响

        out_path = os.path.join('./data', 'lime_pressure')
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        if pattern == 'SST':
            window_size = 256
            data_pattern = 0
        elif pattern == 'DST':
            window_size = 512
            data_pattern = 1

        stride_size = 16

        if name in ['H_P000-0004', 'H_P000-0024']:
            if data_pattern == 0:
                time_date = '2024_05_07_16_32_46'
            elif data_pattern == 1:
                time_date = '2024_05_07_10_14_08'
        elif name in ['H_P000-0033']:
            if data_pattern == 0:
                time_date = '2024_05_07_16_10_46'
            elif data_pattern == 1:
                time_date = '2024_05_07_09_29_04'
        elif name in ['P_09100003']:
            if data_pattern == 0:
                time_date = '2024_05_07_16_18_59'
            elif data_pattern == 1:
                time_date = '2024_05_07_09_42_14'

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_path = os.path.join('./log_dir', time_date, 'checkpoints/best_model/PointNet_cls.pth')

        if model_name == 'PointNet':
            model = PointNet()
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
        model.eval()

        temp_data, L = data_reading(os.path.join('data/raw_data', data_path), dataset, data_pattern)
        pc = temp_data[:, [0, 1, 15]]

        loc_pre = batch_predict(pc, window_size, stride_size, model_name, label_type, device, model)

        explainer = xai_pointcloud.XaiPointcloudExplainer(random_state=2, verbose=False)
        explanation = explainer.explain_instance(pc,
                                                 y_ids=influ_name,
                                                 classifier_fn=batch_predict,
                                                 window_size=window_size,
                                                 stride_size=stride_size,
                                                 model_name=model_name,
                                                 segment_fn=segment_fn,
                                                 num_samples=neighbor_num,
                                                 label_type=label_type,
                                                 device=device,
                                                 model=model,
                                                 )

        mask, metrics = explanation.get_weight_and_shap(1, method)
        colors_map = saliency_map(mask)
        normalized_pc = pc_normalize_all(pc)
        plot_result(normalized_pc, colors_map, mask, out_path, name, data_pattern, if_save=True)

        mask = mask.reshape(mask.shape[0], )
        percentages = np.arange(0.1, 1.1, 0.1)
        positive_values = []
        negative_values = []
        _, fudged_data = segment_fn(pc)

        for attribute in ['positive', 'negative']:
            for percentage in percentages:
                if attribute == 'positive':
                    indices = np.where(mask > 0)[0]
                    sorted_indices = indices[np.lexsort((indices, mask[indices]))]
                    num_mask = int(len(sorted_indices) * percentage)
                    mask_indices = sorted_indices[-num_mask:]
                else:
                    indices = np.where(mask < 0)[0]
                    sorted_indices = indices[np.lexsort((indices, mask[indices]))]
                    num_mask = int(len(sorted_indices) * percentage)
                    mask_indices = sorted_indices[:num_mask]

                reversed_data = deepcopy(pc)
                reversed_data[mask_indices, influ_name] = fudged_data[mask_indices, influ_name]

                mask_pre = batch_predict(reversed_data, window_size, stride_size, model_name, label_type, device, model)

                if attribute == 'positive':
                    positive_values.append((loc_pre[1] - mask_pre[1]) / loc_pre[1])
                else:
                    negative_values.append((loc_pre[1] - mask_pre[1]) / loc_pre[1])

        plt.figure(figsize=(12, 8))
        plt.plot(percentages, positive_values, marker='o', markersize=10, color='darkred', label='Positive', linewidth=5)
        plt.plot(percentages, negative_values, marker='o', markersize=10, color='darkblue', label='Negative', linewidth=5)
        plt.xlabel('Percentage', fontsize=24)
        plt.ylabel('Influence', fontsize=24)
        plt.grid(True)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.savefig(os.path.join(out_path, f'{name}-{str(data_pattern)}-influ.jpg'), dpi=500)
        plt.close()

        return metrics














class GUI_Main(QMainWindow, Ui_MainWindow):
    '''
    主界面
    '''
    def __init__(self, parent=None):
        super(GUI_Main, self).__init__(parent)
        self.setupUi(self)
        self.initUI()
        self.movie = None
        self.paused = False
        self.worker = None
        sys.stdout = Stream(newText=self.onUpdateText)

    def onUpdateText(self, text):
        """Write console output to text widget."""
        cursor = self.textBrowser_2.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser_2.setTextCursor(cursor)
        self.textBrowser_2.ensureCursorVisible()

    def closeEvent(self, event):
        """Shuts down application on close."""
        # Return stdout to defaults.
        if self.movie:
            self.movie.stop()  # 在窗口关闭时停止 GIF 播放
        sys.stdout = sys.__stdout__
        super().closeEvent(event)


    def initUI(self):

        self.pushButton.clicked.connect(self.reproduction) # 复现手绘过程
        self.pushButton_2.clicked.connect(self.diagnosis)  # 诊断过程
        self.pushButton_3.clicked.connect(self.startExplanation)  # 扰动解释

        # 连接按钮到相应的槽函数
        self.pushButton_4.clicked.connect(self.toggle_gif_playback)



    def reproduction(self):
        name = self.comboBox.currentText()
        pattern = self.comboBox_2.currentText()
        dataset = 'ParkinsonHW'
        data_path = f'PD/{name}.txt'

        out_path = os.path.join('./data', 'lime_pressure')
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        if pattern == 'SST':
            data_pattern = 0
        elif pattern == 'DST':
            data_pattern = 1

        temp_data, L = data_reading(os.path.join('data/raw_data', data_path), dataset, data_pattern)
        pc = temp_data[:, [0, 1, 15]]
        matplotlib_gif(pc, out_path, name, data_pattern)

        gif_name = f'{name}-{str(data_pattern)}.gif'
        gif_path = os.path.join(out_path, gif_name)

        self.show_gif(gif_path)

    def show_gif(self, gif_path):
        self.movie = QMovie(gif_path)
        if self.movie.isValid():
            self.label_9.setMovie(self.movie)
            self.movie.setScaledSize(self.label_9.size())
            self.movie.start()
        else:
            QMessageBox.warning(self, "提示", "GIF文件加载失败！")

    def toggle_gif_playback(self):
        if self.movie:
            if self.paused:
                self.movie.start()
                self.pushButton_4.setText("Pause")
                self.paused = False
            else:
                self.movie.setPaused(True)
                self.pushButton_4.setText("Continue")
                self.paused = True



    def diagnosis(self):
        name = self.comboBox.currentText()
        pattern = self.comboBox_2.currentText()
        model_name = self.comboBox_3.currentText()

        dataset = 'ParkinsonHW'
        data_path = f'PD/{name}.txt'

        label_type = data_path.strip().split('/')[0]

        out_path = os.path.join('./data', 'lime_pressure')
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        if pattern == 'SST':
            window_size = 256
            data_pattern = 0
        elif pattern == 'DST':
            window_size = 512
            data_pattern = 1

        stride_size = 16


        if name in ['H_P000-0004', 'H_P000-0024']:
            if data_pattern == 0:
                time_date = '2024_05_07_16_32_46'
            elif data_pattern == 1:
                time_date = '2024_05_07_10_14_08'
        elif name in ['H_P000-0033']:
            if data_pattern == 0:
                time_date = '2024_05_07_16_10_46'
            elif data_pattern == 1:
                time_date = '2024_05_07_09_29_04'
        elif name in ['P_09100003']:
            if data_pattern == 0:
                time_date = '2024_05_07_16_18_59'
            elif data_pattern == 1:
                time_date = '2024_05_07_09_42_14'
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_path = os.path.join('./log_dir', time_date, 'checkpoints/best_model/PointNet_cls.pth')
        if model_name == 'PointNet':
            model = PointNet()
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
        model.eval()

        temp_data, L = data_reading(os.path.join('data/raw_data', data_path), dataset, data_pattern)
        pc = temp_data[:, [0, 1, 15]]

        loc_pre = batch_predict(pc, window_size, stride_size, model_name, label_type, device, model)

        matplotlib_result(loc_pre, out_path, name, data_pattern)

        jpg_name = f'{name}-{str(data_pattern)}.jpg'
        jpg_path = os.path.join(out_path, jpg_name)
        pixmap = QPixmap(jpg_path)
        self.label_10.setPixmap(pixmap)
        self.label_10.setScaledContents(True)




    def startExplanation(self):

        name = self.comboBox.currentText()
        pattern = self.comboBox_2.currentText()
        model_name = self.comboBox_3.currentText()
        neighbor_num = int(self.comboBox_4.currentText())

        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Warning", "Explanation is already running. Please wait.")
            return

        self.worker = ExplanationWorker()
        self.worker.set_parameters(name, pattern, model_name, neighbor_num)
        self.worker.finished.connect(self.onExplanationFinished)
        self.worker.resultReady.connect(self.onResultReady)  # 连接结果信号
        self.worker.start()

    def onExplanationFinished(self):
        # 耗时操作完成后的处理，例如更新界面
        QMessageBox.information(self, "Information", "Explanation finished successfully.")
        # 更新界面其他部分，例如显示图片等

    def onResultReady(self, metrics):
        name = self.comboBox.currentText()
        pattern = self.comboBox_2.currentText()
        if pattern == 'SST':
            window_size = 256
            data_pattern = 0
        elif pattern == 'DST':
            window_size = 512
            data_pattern = 1
        out_path = os.path.join('./data', 'lime_pressure')

        jpg_exp_name = f'{name}-{str(data_pattern)}-explain.jpg'
        jpg_exp_path = os.path.join(out_path, jpg_exp_name)
        pixmap = QPixmap(jpg_exp_path)
        self.label_11.setPixmap(pixmap)
        self.label_11.setScaledContents(True)


        jpg_inf_name = f'{name}-{str(data_pattern)}-influ.jpg'
        jpg_inf_path = os.path.join(out_path, jpg_inf_name)
        pixmap_1 = QPixmap(jpg_inf_path)
        self.label_12.setPixmap(pixmap_1)
        self.label_12.setScaledContents(True)

        mae = metrics[1]
        mae_str = f"{mae:.6f}"
        self.lineEdit.setText(mae_str)

        mse = metrics[3]
        mse_str = f"{mse:.6f}"
        self.lineEdit_2.setText(mse_str)

        evs = metrics[7]
        evs_str = f"{evs:.6f}"
        self.lineEdit_3.setText(evs_str)

        r2 = metrics[9]
        r2_str = f"{r2:.6f}"
        self.lineEdit_4.setText(r2_str)



if __name__=="__main__":
    # QGuiApplication.setAttribute(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    main_win = GUI_Main()  # 主界面
    main_win.show()
    sys.exit(app.exec_())