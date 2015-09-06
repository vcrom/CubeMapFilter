QT += core
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
TEMPLATE = app

CONFIG += console
CONFIG -= app_bundle
CONFIG += c++11

win32|win64 {
# OPENCV_DIR is the env variable used by the official OpenCV documentation
INCLUDEPATH += $$(OPENCV_DIR)/../../include

QMAKE_CXXFLAGS += -D_USE_MATH_DEFINES # for M_PI

LIBS += -L$$(OPENCV_DIR)/lib
LIBS += -lopencv_world300
}
else {
LIBS += -lopencv_core
LIBS += -lopencv_highgui
LIBS += -lopencv_imgproc
}
QMAKE_CXXFLAGS_RELEASE += -O3 -fopenmp
Release:LIBS += -fopenmp

SOURCES += main.cpp \
    cubemap.cpp \
    utils.cpp \
    cubemapfilter.cpp

HEADERS += \
    cubemap.h \
    utils.h \
    cubemapfilter.h
