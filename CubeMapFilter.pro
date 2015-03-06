
QT       += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
TEMPLATE = app


CONFIG += console
CONFIG -= app_bundle
CONFIG += c++11

SOURCES += main.cpp \
    cubemap.cpp \
    utils.cpp \
    cubemapfilter.cpp

LIBS += -lopencv_core
LIBS += -lopencv_highgui
LIBS += -lopencv_imgproc

HEADERS += \
    cubemap.h \
    utils.h \
    cubemapfilter.h

QMAKE_CXXFLAGS    += -O3
QMAKE_CXXFLAGS += -fopenmp
LIBS += -fopenmp
