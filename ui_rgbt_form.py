# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'rgbt_form.ui'
##
## Created by: Qt User Interface Compiler version 6.2.4
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QSlider,
    QTabWidget, QWidget)

class Ui_RGBT(object):
    def setupUi(self, RGBT):
        if not RGBT.objectName():
            RGBT.setObjectName(u"RGBT")
        RGBT.resize(1400, 888)
        RGBT.setMinimumSize(QSize(0, 0))
        RGBT.setMaximumSize(QSize(1400, 1000))
        self.gridLayout = QGridLayout(RGBT)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label = QLabel(RGBT)
        self.label.setObjectName(u"label")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMaximumSize(QSize(1380, 70))
        font = QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setPixmap(QPixmap(u"images/banner.png"))
        self.label.setScaledContents(True)
        self.label.setWordWrap(True)

        self.gridLayout.addWidget(self.label, 0, 0, 1, 2)

        self.log_label = QLabel(RGBT)
        self.log_label.setObjectName(u"log_label")
        self.log_label.setMaximumSize(QSize(1280, 30))

        self.gridLayout.addWidget(self.log_label, 4, 0, 1, 2)

        self.groupBox_3 = QGroupBox(RGBT)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setMaximumSize(QSize(680, 550))
        font1 = QFont()
        font1.setPointSize(12)
        self.groupBox_3.setFont(font1)
        self.pix_label_rgb = QLabel(self.groupBox_3)
        self.pix_label_rgb.setObjectName(u"pix_label_rgb")
        self.pix_label_rgb.setGeometry(QRect(10, 30, 650, 501))
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.pix_label_rgb.sizePolicy().hasHeightForWidth())
        self.pix_label_rgb.setSizePolicy(sizePolicy1)
        self.pix_label_rgb.setMaximumSize(QSize(650, 520))
        self.pix_label_rgb.setFont(font1)

        self.gridLayout.addWidget(self.groupBox_3, 2, 0, 1, 1)

        self.groupBox_2 = QGroupBox(RGBT)
        self.groupBox_2.setObjectName(u"groupBox_2")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy2)
        self.groupBox_2.setMinimumSize(QSize(0, 0))
        self.groupBox_2.setMaximumSize(QSize(680, 550))
        self.groupBox_2.setFont(font1)
        self.horizontalLayout = QHBoxLayout(self.groupBox_2)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.pix_label = QLabel(self.groupBox_2)
        self.pix_label.setObjectName(u"pix_label")
        sizePolicy1.setHeightForWidth(self.pix_label.sizePolicy().hasHeightForWidth())
        self.pix_label.setSizePolicy(sizePolicy1)
        self.pix_label.setMinimumSize(QSize(0, 0))
        self.pix_label.setMaximumSize(QSize(650, 520))
        self.pix_label.setFont(font1)
        self.pix_label.setScaledContents(True)

        self.horizontalLayout.addWidget(self.pix_label)


        self.gridLayout.addWidget(self.groupBox_2, 2, 1, 1, 1)

        self.groupBox = QGroupBox(RGBT)
        self.groupBox.setObjectName(u"groupBox")
        sizePolicy2.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy2)
        self.groupBox.setMinimumSize(QSize(0, 0))
        self.groupBox.setMaximumSize(QSize(1380, 220))
        self.groupBox.setFont(font1)
        self.gridLayout_2 = QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.tabWidget = QTabWidget(self.groupBox)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setMaximumSize(QSize(1370, 250))
        self.tabWidget.setFont(font1)
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.gridLayout_3 = QGridLayout(self.tab)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.comboBox_RGB_Cam = QComboBox(self.tab)
        self.comboBox_RGB_Cam.addItem("")
        self.comboBox_RGB_Cam.addItem("")
        self.comboBox_RGB_Cam.addItem("")
        self.comboBox_RGB_Cam.addItem("")
        self.comboBox_RGB_Cam.addItem("")
        self.comboBox_RGB_Cam.setObjectName(u"comboBox_RGB_Cam")
        self.comboBox_RGB_Cam.setEnabled(False)
        sizePolicy2.setHeightForWidth(self.comboBox_RGB_Cam.sizePolicy().hasHeightForWidth())
        self.comboBox_RGB_Cam.setSizePolicy(sizePolicy2)
        self.comboBox_RGB_Cam.setStyleSheet(u"text-align:left")
        self.comboBox_RGB_Cam.setMaxVisibleItems(10)

        self.gridLayout_3.addWidget(self.comboBox_RGB_Cam, 1, 1, 1, 1)

        self.checkBox_Th = QCheckBox(self.tab)
        self.checkBox_Th.setObjectName(u"checkBox_Th")
        sizePolicy2.setHeightForWidth(self.checkBox_Th.sizePolicy().hasHeightForWidth())
        self.checkBox_Th.setSizePolicy(sizePolicy2)

        self.gridLayout_3.addWidget(self.checkBox_Th, 0, 0, 1, 1)

        self.connectButton_Thermal = QPushButton(self.tab)
        self.connectButton_Thermal.setObjectName(u"connectButton_Thermal")
        self.connectButton_Thermal.setEnabled(False)
        sizePolicy2.setHeightForWidth(self.connectButton_Thermal.sizePolicy().hasHeightForWidth())
        self.connectButton_Thermal.setSizePolicy(sizePolicy2)
        self.connectButton_Thermal.setFont(font1)
        self.connectButton_Thermal.setStyleSheet(u"text-align:left")

        self.gridLayout_3.addWidget(self.connectButton_Thermal, 0, 1, 1, 1)

        self.label_6 = QLabel(self.tab)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout_3.addWidget(self.label_6, 2, 6, 1, 1)

        self.checkBox_RGB = QCheckBox(self.tab)
        self.checkBox_RGB.setObjectName(u"checkBox_RGB")
        sizePolicy2.setHeightForWidth(self.checkBox_RGB.sizePolicy().hasHeightForWidth())
        self.checkBox_RGB.setSizePolicy(sizePolicy2)

        self.gridLayout_3.addWidget(self.checkBox_RGB, 1, 0, 1, 1)

        self.comboBox_focus = QComboBox(self.tab)
        self.comboBox_focus.addItem("")
        self.comboBox_focus.addItem("")
        self.comboBox_focus.setObjectName(u"comboBox_focus")
        sizePolicy2.setHeightForWidth(self.comboBox_focus.sizePolicy().hasHeightForWidth())
        self.comboBox_focus.setSizePolicy(sizePolicy2)
        self.comboBox_focus.setStyleSheet(u"text-align:left")
        self.comboBox_focus.setMaxVisibleItems(2)

        self.gridLayout_3.addWidget(self.comboBox_focus, 2, 7, 1, 1)

        self.comboBox_fps = QComboBox(self.tab)
        self.comboBox_fps.addItem("")
        self.comboBox_fps.addItem("")
        self.comboBox_fps.addItem("")
        self.comboBox_fps.addItem("")
        self.comboBox_fps.addItem("")
        self.comboBox_fps.addItem("")
        self.comboBox_fps.addItem("")
        self.comboBox_fps.addItem("")
        self.comboBox_fps.setObjectName(u"comboBox_fps")
        sizePolicy2.setHeightForWidth(self.comboBox_fps.sizePolicy().hasHeightForWidth())
        self.comboBox_fps.setSizePolicy(sizePolicy2)
        self.comboBox_fps.setStyleSheet(u"text-align:left")
        self.comboBox_fps.setMaxVisibleItems(5)

        self.gridLayout_3.addWidget(self.comboBox_fps, 1, 7, 1, 1)

        self.comboBox_vis = QComboBox(self.tab)
        self.comboBox_vis.addItem("")
        self.comboBox_vis.addItem("")
        self.comboBox_vis.addItem("")
        self.comboBox_vis.addItem("")
        self.comboBox_vis.addItem("")
        self.comboBox_vis.addItem("")
        self.comboBox_vis.addItem("")
        self.comboBox_vis.addItem("")
        self.comboBox_vis.addItem("")
        self.comboBox_vis.addItem("")
        self.comboBox_vis.addItem("")
        self.comboBox_vis.setObjectName(u"comboBox_vis")
        sizePolicy2.setHeightForWidth(self.comboBox_vis.sizePolicy().hasHeightForWidth())
        self.comboBox_vis.setSizePolicy(sizePolicy2)
        self.comboBox_vis.setStyleSheet(u"text-align:left")

        self.gridLayout_3.addWidget(self.comboBox_vis, 0, 7, 1, 1)

        self.label_7 = QLabel(self.tab)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout_3.addWidget(self.label_7, 0, 6, 1, 1)

        self.label_8 = QLabel(self.tab)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout_3.addWidget(self.label_8, 1, 6, 1, 1)

        self.label_pid = QLabel(self.tab)
        self.label_pid.setObjectName(u"label_pid")

        self.gridLayout_3.addWidget(self.label_pid, 3, 0, 1, 1)

        self.text_pid = QLineEdit(self.tab)
        self.text_pid.setObjectName(u"text_pid")
        sizePolicy2.setHeightForWidth(self.text_pid.sizePolicy().hasHeightForWidth())
        self.text_pid.setSizePolicy(sizePolicy2)
        self.text_pid.setStyleSheet(u"text-align:left")

        self.gridLayout_3.addWidget(self.text_pid, 3, 1, 1, 1)

        self.label_2 = QLabel(self.tab)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout_3.addWidget(self.label_2, 2, 0, 1, 1)

        self.text_study = QLineEdit(self.tab)
        self.text_study.setObjectName(u"text_study")
        sizePolicy2.setHeightForWidth(self.text_study.sizePolicy().hasHeightForWidth())
        self.text_study.setSizePolicy(sizePolicy2)

        self.gridLayout_3.addWidget(self.text_study, 2, 1, 1, 1)

        self.line = QFrame(self.tab)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.gridLayout_3.addWidget(self.line, 0, 4, 5, 1)

        self.acquireButton = QPushButton(self.tab)
        self.acquireButton.setObjectName(u"acquireButton")
        self.acquireButton.setEnabled(False)
        sizePolicy2.setHeightForWidth(self.acquireButton.sizePolicy().hasHeightForWidth())
        self.acquireButton.setSizePolicy(sizePolicy2)
        self.acquireButton.setFont(font1)
        self.acquireButton.setStyleSheet(u"text-align:center")

        self.gridLayout_3.addWidget(self.acquireButton, 4, 0, 1, 2)

        self.recordButton = QPushButton(self.tab)
        self.recordButton.setObjectName(u"recordButton")
        self.recordButton.setEnabled(False)
        sizePolicy2.setHeightForWidth(self.recordButton.sizePolicy().hasHeightForWidth())
        self.recordButton.setSizePolicy(sizePolicy2)
        self.recordButton.setFont(font1)
        self.recordButton.setStyleSheet(u"text-align:center")

        self.gridLayout_3.addWidget(self.recordButton, 4, 6, 1, 2)

        self.horizontalSlider_focus = QSlider(self.tab)
        self.horizontalSlider_focus.setObjectName(u"horizontalSlider_focus")
        sizePolicy2.setHeightForWidth(self.horizontalSlider_focus.sizePolicy().hasHeightForWidth())
        self.horizontalSlider_focus.setSizePolicy(sizePolicy2)
        self.horizontalSlider_focus.setMaximum(255)
        self.horizontalSlider_focus.setSingleStep(5)
        self.horizontalSlider_focus.setPageStep(20)
        self.horizontalSlider_focus.setValue(10)
        self.horizontalSlider_focus.setSliderPosition(10)
        self.horizontalSlider_focus.setOrientation(Qt.Horizontal)

        self.gridLayout_3.addWidget(self.horizontalSlider_focus, 3, 7, 1, 1)

        self.label_4 = QLabel(self.tab)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout_3.addWidget(self.label_4, 3, 6, 1, 1)

        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.gridLayout_4 = QGridLayout(self.tab_2)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.pushButton_2 = QPushButton(self.tab_2)
        self.pushButton_2.setObjectName(u"pushButton_2")

        self.gridLayout_4.addWidget(self.pushButton_2, 1, 0, 1, 2)

        self.browseButton = QPushButton(self.tab_2)
        self.browseButton.setObjectName(u"browseButton")

        self.gridLayout_4.addWidget(self.browseButton, 0, 0, 1, 1)

        self.label_3 = QLabel(self.tab_2)
        self.label_3.setObjectName(u"label_3")
        font2 = QFont()
        font2.setPointSize(12)
        font2.setItalic(True)
        self.label_3.setFont(font2)

        self.gridLayout_4.addWidget(self.label_3, 0, 1, 1, 1)

        self.tabWidget.addTab(self.tab_2, "")

        self.gridLayout_2.addWidget(self.tabWidget, 0, 1, 1, 1)


        self.gridLayout.addWidget(self.groupBox, 1, 0, 1, 2)


        self.retranslateUi(RGBT)

        self.tabWidget.setCurrentIndex(0)
        self.comboBox_RGB_Cam.setCurrentIndex(0)
        self.comboBox_focus.setCurrentIndex(0)
        self.comboBox_fps.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(RGBT)
    # setupUi

    def retranslateUi(self, RGBT):
        RGBT.setWindowTitle(QCoreApplication.translate("RGBT", u"TIComp", None))
        self.label.setText("")
        self.log_label.setText(QCoreApplication.translate("RGBT", u"Info Log", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("RGBT", u"RGB Image Streaming", None))
        self.pix_label_rgb.setText(QCoreApplication.translate("RGBT", u"RGB Image streaming will appear here", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("RGBT", u"Thermal Image Streaming", None))
        self.pix_label.setText(QCoreApplication.translate("RGBT", u"Thermal Image streaming will appear here", None))
        self.groupBox.setTitle(QCoreApplication.translate("RGBT", u"Controls", None))
        self.comboBox_RGB_Cam.setItemText(0, QCoreApplication.translate("RGBT", u"0", None))
        self.comboBox_RGB_Cam.setItemText(1, QCoreApplication.translate("RGBT", u"1", None))
        self.comboBox_RGB_Cam.setItemText(2, QCoreApplication.translate("RGBT", u"2", None))
        self.comboBox_RGB_Cam.setItemText(3, QCoreApplication.translate("RGBT", u"3", None))
        self.comboBox_RGB_Cam.setItemText(4, QCoreApplication.translate("RGBT", u"4", None))

#if QT_CONFIG(tooltip)
        self.comboBox_RGB_Cam.setToolTip(QCoreApplication.translate("RGBT", u"<html><head/><body><p>Select RGB Camera</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.checkBox_Th.setText(QCoreApplication.translate("RGBT", u"Capture Thermal", None))
        self.connectButton_Thermal.setText(QCoreApplication.translate("RGBT", u"Scan and Connect", None))
        self.label_6.setText(QCoreApplication.translate("RGBT", u"Focus Type", None))
        self.checkBox_RGB.setText(QCoreApplication.translate("RGBT", u"Capture RGB", None))
        self.comboBox_focus.setItemText(0, QCoreApplication.translate("RGBT", u"Auto", None))
        self.comboBox_focus.setItemText(1, QCoreApplication.translate("RGBT", u"Manual", None))

#if QT_CONFIG(tooltip)
        self.comboBox_focus.setToolTip(QCoreApplication.translate("RGBT", u"<html><head/><body><p>Focus Type</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.comboBox_fps.setItemText(0, QCoreApplication.translate("RGBT", u"1", None))
        self.comboBox_fps.setItemText(1, QCoreApplication.translate("RGBT", u"2", None))
        self.comboBox_fps.setItemText(2, QCoreApplication.translate("RGBT", u"5", None))
        self.comboBox_fps.setItemText(3, QCoreApplication.translate("RGBT", u"10", None))
        self.comboBox_fps.setItemText(4, QCoreApplication.translate("RGBT", u"15", None))
        self.comboBox_fps.setItemText(5, QCoreApplication.translate("RGBT", u"25", None))
        self.comboBox_fps.setItemText(6, QCoreApplication.translate("RGBT", u"30", None))
        self.comboBox_fps.setItemText(7, QCoreApplication.translate("RGBT", u"60", None))

        self.comboBox_vis.setItemText(0, QCoreApplication.translate("RGBT", u"CoolWarm", None))
        self.comboBox_vis.setItemText(1, QCoreApplication.translate("RGBT", u"GNUPlot2", None))
        self.comboBox_vis.setItemText(2, QCoreApplication.translate("RGBT", u"Gray", None))
        self.comboBox_vis.setItemText(3, QCoreApplication.translate("RGBT", u"Magma", None))
        self.comboBox_vis.setItemText(4, QCoreApplication.translate("RGBT", u"Nipy_Spectral", None))
        self.comboBox_vis.setItemText(5, QCoreApplication.translate("RGBT", u"Pink", None))
        self.comboBox_vis.setItemText(6, QCoreApplication.translate("RGBT", u"Plasma", None))
        self.comboBox_vis.setItemText(7, QCoreApplication.translate("RGBT", u"Prism", None))
        self.comboBox_vis.setItemText(8, QCoreApplication.translate("RGBT", u"Rainbow", None))
        self.comboBox_vis.setItemText(9, QCoreApplication.translate("RGBT", u"Seismic", None))
        self.comboBox_vis.setItemText(10, QCoreApplication.translate("RGBT", u"Terrain", None))

        self.label_7.setText(QCoreApplication.translate("RGBT", u"Pseudo-color Palette for Visualization", None))
        self.label_8.setText(QCoreApplication.translate("RGBT", u"Frame Rate", None))
        self.label_pid.setText(QCoreApplication.translate("RGBT", u"Participant ID", None))
        self.label_2.setText(QCoreApplication.translate("RGBT", u"Study Name", None))
        self.acquireButton.setText(QCoreApplication.translate("RGBT", u"Start Live Streaming", None))
        self.recordButton.setText(QCoreApplication.translate("RGBT", u"Start Recording", None))
#if QT_CONFIG(tooltip)
        self.horizontalSlider_focus.setToolTip(QCoreApplication.translate("RGBT", u"<html><head/><body><p>0 to 255, in increment of 5; </p><p>0 indicates focus at infinity. 255 indicates nearest focus.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.label_4.setText(QCoreApplication.translate("RGBT", u"Focus Adjustment", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("RGBT", u"Live Acquisition", None))
        self.pushButton_2.setText(QCoreApplication.translate("RGBT", u"Start Playing", None))
        self.browseButton.setText(QCoreApplication.translate("RGBT", u"Browse", None))
        self.label_3.setText(QCoreApplication.translate("RGBT", u"Directory Path", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("RGBT", u"Recorded Data", None))
    # retranslateUi

