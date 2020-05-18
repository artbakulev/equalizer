import threading
import wave
from math import (sin, pi)
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pygame
from librosa import load
from PyQt5.QtCore import (Qt, QThreadPool)
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                             QFileDialog,
                             QSlider, QPushButton, QLabel,
                             QCheckBox, QLCDNumber)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from pysndfx import AudioEffectsChain

from scipy.signal import cheby1, filtfilt
from app.Worker import Worker


class IntTypes:
    types = {
        1: np.int8,
        2: np.int16,
        4: np.int32
    }


class Main_Window_class(QDialog):
    PLOTS_NUMBER = 3
    EFFECTS_NUMBER = 2

    def __init__(self):
        super().__init__()
        self.nlabels = 8

        self.filter_order = 10

        # показатель пульсаций определяет амплитуду пульсаций
        self.riple_factor = 0.5

        # Wn is a fraction of the Nyquist frequency (half the sampling frequency). So if the sampling rate is 1000Hz
        # and you want a cutoff of 250Hz, you should use Wn=0.5.
        self.Wn = 0.5

        self.slider_step = 1
        self.slider_default = 0
        self.slider_max = 50
        self.slider_min = -50

        self.music_is_playing = False
        self.threadpool = QThreadPool()
        self.nchannels = None  # number of channels
        self.sampwidth = None  # number of bytes per sample
        self.framerate = None  # number of frames per second
        self.nframes = None  # total number of frames
        self.comptype = None  # compression type
        self.compname = None  # compression type name
        self.elem_per_hertz = None
        self.coefficient = 1000  # коэффициент прореживания, для уменьшения частоты дискретизации
        self.buffer_size = None
        self.buffer_cnt = 0
        self.music_worker = None
        self.min_freq = 0
        self.max_freq = None

        self.channels = []
        self.spectrum = None

        self.spectrum_original = None
        self.channels_original = []

        self.spectrum_reverberation = None
        self.channels_reverberation = []

        self.spectrum_vibrato = None
        self.channels_vibrato = []
        self.path_to_pull = None

        self.app_name = 'Эквалайзер'
        self.buttons_labels = ['Воспроизвести', 'Пауза', 'Остановить']
        self.bands = [[], []]
        self.labels = []
        self.ui_labels = []
        self.sliders = []
        self.sliders_workers = [None for _ in range(self.nlabels)]
        self.sliders_old_values = [self.slider_default for _ in range(self.nlabels)]
        self.LCD_numbers = []
        self.canvases = []
        self.effects = [self.doing_reverberation, self.doing_vibrato]
        self.effects_checkboxes_labels = ['Реверберация', 'Вибрато']
        self.effects_checkboxes = []
        self.effects_checkboxes_workers = []
        self.play_button, self.stop_button, self.pause_button = None, None, None
        self.redraw_mutex = threading.Lock()

        self.run_ui()

    def run_ui(self):
        self.pull_music()
        self.create_interface()

    def pull_music(self):
        self.path_to_pull = QFileDialog.getOpenFileName(self, 'Выберите .wav файл')[0]
        wav = wave.open(self.path_to_pull, mode='r')

        # nchannels - число каналов .wav файла
        # sampwidth - число байт на сэмпл
        # framerate - число фреймов в секунду
        # nframes - общее число фреймов
        # comptype - тип сжатия
        # compname - имя типа сжатия
        (self.nchannels, self.sampwidth,
         self.framerate, self.nframes,
         self.comptype, self.compname) = wav.getparams()

        # Теорема Котельникова, в дискретном сигнале представлены частоты от нуля до framerate // 2
        self.max_freq = self.framerate // 2
        self.buffer_size = self.framerate

        # считываем все фреймы
        content = wav.readframes(self.nframes)

        # и сохраняем в массив с нужным типом элемента
        samples = np.fromstring(content, dtype=IntTypes.types[self.sampwidth])

        # разбиваем на каналы, например, \xe2\xff\xe3\xfа — это фрейм 16-битного wav-файла. Значит, \xe2\xff — сэмпл
        # первого (левого) канала, а \xe3\xfа
        for i in range(self.nchannels):
            self.channels.append(samples[i::self.nchannels])

        # сохраняем оригинальные каналы
        self.channels_original = self.channels.copy()
        # self.channels = self.cheby1_filtration(self.channels)

        # создаем worker'ов, который создадут измененные дорожки с примененными эффектами
        for i in range(self.EFFECTS_NUMBER):
            worker = Worker(self.effects[i], self.channels)
            self.effects_checkboxes_workers.append(worker)
            self.threadpool.start(worker)

        # Выполняем быстрое дискретное преобразование Фурье, для получения спектра
        self.spectrum = np.fft.rfft(self.channels_original)
        self.spectrum_original = self.spectrum.copy()

        # запускаем pygame
        pygame.mixer.pre_init(frequency=self.framerate,
                              size=-8 * self.sampwidth,
                              channels=self.nchannels)
        pygame.init()

    def create_bands(self):
        """Создает подписи к слайдерам"""
        step = (self.max_freq - self.min_freq) // 2 ** self.nlabels

        self.bands[0].append(self.min_freq)
        self.bands[1].append(self.min_freq + step)

        for i in range(1, self.nlabels - 1):
            self.bands[0].append(self.bands[1][i - 1])
            self.bands[1].append(self.bands[0][i] + 2 ** i * step)

        self.bands[0].append(self.bands[1][self.nlabels - 2])
        self.bands[1].append(self.max_freq)

        for i in range(self.nlabels):
            self.labels.append(f'{self.bands[0][i]} - {str(self.bands[1][i])}')

    def create_labels(self):
        for label in self.labels:
            self.ui_labels.append(QLabel(label, self))

    def create_lcd_numbers(self):
        for _ in range(self.nlabels):
            self.LCD_numbers.append(QLCDNumber(self))

    def create_sliders(self):
        for i in range(self.nlabels):
            slider = QSlider(Qt.Vertical, self)
            slider.setMinimum(self.slider_min)
            slider.setMaximum(self.slider_max)
            slider.setValue(self.slider_default)
            slider.setFocusPolicy(Qt.StrongFocus)
            slider.setTickPosition(QSlider.TicksBothSides)
            slider.setSingleStep(self.slider_step)
            slider.valueChanged[int].connect(self.slider_change_value)
            slider.valueChanged[int].connect(self.LCD_numbers[i].display)
            self.sliders.append(slider)

    def create_checkboxes(self):
        for i in range(self.EFFECTS_NUMBER):
            checkbox = QCheckBox(self.effects_checkboxes_labels[i], self)
            checkbox.setChecked(False)
            checkbox.stateChanged.connect(self.checkbox_clicked_listener)
            self.effects_checkboxes.append(checkbox)

    def create_buttons(self):
        self.play_button = QPushButton(self.buttons_labels[0], self)
        self.pause_button = QPushButton(self.buttons_labels[1], self)
        self.stop_button = QPushButton(self.buttons_labels[2], self)

        self.play_button.clicked.connect(self.button_clicked_listener)
        self.pause_button.clicked.connect(self.button_clicked_listener)
        self.stop_button.clicked.connect(self.button_clicked_listener)

    def create_graphics(self):
        self.create_lcd_numbers()
        self.create_sliders()
        self.create_bands()
        self.create_labels()
        self.create_checkboxes()
        self.create_buttons()

        self.elem_per_hertz = self.spectrum.shape[1] // (self.max_freq - self.min_freq)
        plots_labels = [('Частота, Гц', 'Амплитуда'), ('Частота, Гц', 'Амплитуда'), ('Время, с', 'Амплитуда')]
        plots_sources = [(self.channels[0][::self.coefficient],),
                         (np.fft.rfftfreq(self.nframes, 1. / self.framerate)[::self.coefficient],
                          np.abs(self.spectrum[0][::self.coefficient]) / self.nframes),
                         (np.fft.rfftfreq(self.nframes, 1. / self.framerate)[::self.coefficient],
                          np.abs(self.spectrum[0][::self.coefficient]) / self.nframes)]
        for i in range(self.PLOTS_NUMBER):
            figure = plt.figure()
            subplot = figure.add_subplot(1, 1, 1)
            subplot.plot(*plots_sources[i])
            subplot.set_xlabel(plots_labels[i][0])
            subplot.set_ylabel(plots_labels[i][1])
            figure.align_xlabels()
            figure.align_ylabels()
            canvas = FigureCanvas(figure)
            navigation_tool = NavigationToolbar(canvas, self)
            canvas = {
                'figure': figure,
                'canvas': canvas,
                'navigation_tool': navigation_tool,
                'axes_labels': plots_labels[i],
            }
            self.canvases.append(canvas)
            canvas['canvas'].draw()

    def create_interface(self):
        self.create_graphics()
        labels_box = QHBoxLayout()
        for label in self.ui_labels:
            labels_box.addWidget(label)

        nums_box = QHBoxLayout()
        for number in self.LCD_numbers:
            nums_box.addWidget(number)

        sliders_box = QHBoxLayout()
        for slider in self.sliders:
            sliders_box.addWidget(slider)

        graph_box = QVBoxLayout()
        graph_box.addWidget(self.canvases[0]['navigation_tool'])
        graph_box.addWidget(self.canvases[0]['canvas'])

        left_box = QVBoxLayout()
        left_box.addLayout(labels_box)
        left_box.addLayout(sliders_box)
        left_box.addLayout(nums_box)
        left_box.addLayout(graph_box)

        checkbox_and_button_layout = QHBoxLayout()
        for i in range(self.EFFECTS_NUMBER):
            checkbox_and_button_layout.addWidget(self.effects_checkboxes[i])
        checkbox_and_button_layout.addWidget(self.play_button)
        checkbox_and_button_layout.addWidget(self.pause_button)
        checkbox_and_button_layout.addWidget(self.stop_button)

        graph_box = QVBoxLayout()
        graph_box.addWidget(self.canvases[1]['navigation_tool'])
        graph_box.addWidget(self.canvases[1]['canvas'])
        graph_box.addWidget(self.canvases[2]['navigation_tool'])
        graph_box.addWidget(self.canvases[2]['canvas'])

        right_box = QVBoxLayout()
        right_box.addLayout(checkbox_and_button_layout)
        right_box.addLayout(graph_box)

        all_box = QHBoxLayout()
        all_box.addLayout(right_box)
        all_box.addLayout(left_box)

        self.setLayout(all_box)

        self.setWindowTitle(self.app_name)
        self.showMaximized()

    def slider_change_value(self, value):
        for i, slider in enumerate(self.sliders):
            if self.sender() == slider:
                self.sliders_workers[i] = Worker(self.music_edit, i, value)
                self.threadpool.start(self.sliders_workers[i])

    def checkbox_clicked_listener(self, state):
        if self.sender() == self.effects_checkboxes[0]:
            if state == Qt.Checked:
                self.effects_checkboxes[1].setChecked(False)
                self.channels = self.channels_reverberation.copy()
                self.spectrum = self.spectrum_reverberation.copy()
            else:
                self.channels = self.channels_original.copy()
                self.spectrum = self.spectrum_original.copy()

        elif self.sender() == self.effects_checkboxes[1]:
            if state == Qt.Checked:
                self.effects_checkboxes[0].setChecked(False)
                self.channels = self.channels_vibrato.copy()
                self.spectrum = self.spectrum_vibrato.copy()
            else:
                self.channels = self.channels_original.copy()
                self.spectrum = self.spectrum_original.copy()

        for slider in self.sliders:
            slider.setValue(self.slider_default)

        draw_1 = Worker(self.draw_array, self.spectrum, 0)
        self.threadpool.start(draw_1)

        draw_2 = Worker(self.draw_array, self.channels, 1)
        self.threadpool.start(draw_2)

    def button_clicked_listener(self):
        if self.sender() == self.play_button:
            # Запустить
            if not self.music_is_playing:
                self.music_is_playing = True
                self.music_worker = Worker(self.start_music)
                self.threadpool.start(self.music_worker)

        elif self.sender() == self.pause_button:
            # Пауза
            if self.music_is_playing:
                self.music_is_playing = False

        elif self.sender() == self.stop_button:
            # Остановить
            if self.music_is_playing:
                self.music_is_playing = False

            self.threadpool.clear()

            for slider in self.sliders:
                worker = Worker(self.music_edit, self.sliders.index(slider), self.slider_default)
                self.sliders_workers.append(worker)
                self.threadpool.start(worker)

            self.threadpool.start(Worker(self.wait_until_sliders_launched))

            for slider in self.sliders:
                slider.setValue(self.slider_default)

            self.buffer_cnt = 0

            for i in range(self.EFFECTS_NUMBER):
                self.effects_checkboxes[i].setChecked(False)

    def wait_until_sliders_launched(self):
        while self.threadpool.activeThreadCount() != len(self.sliders):
            sleep(0.1)
        self.channels = self.channels_original.copy()
        self.spectrum = self.spectrum_original.copy()

    def cheby1_filtration(self, channels):
        # noinspection PyTupleAssignmentBalance
        b, a = cheby1(self.filter_order, self.riple_factor, self.Wn, output='ba')
        channels = filtfilt(b, a, channels)
        return channels

    def start_music(self):
        tmp_channels = [self.channels[0][self.buffer_cnt * self.buffer_size:
                                         (self.buffer_cnt + 1) * self.buffer_size + 1:],
                        self.channels[1][self.buffer_cnt * self.buffer_size:
                                         (self.buffer_cnt + 1) * self.buffer_size + 1:]]
        tmp_channels = np.array(tmp_channels)
        tmp_channels = np.ascontiguousarray(tmp_channels.T)
        tmp_sound = pygame.sndarray.make_sound(tmp_channels)

        sound = tmp_sound
        if not self.music_is_playing:
            return
        pygame.mixer.Sound.play(sound)

        start_pos = self.buffer_cnt
        for self.buffer_cnt in range(start_pos + 1, self.nframes // self.buffer_size):
            tmp_channels = [self.channels[0][self.buffer_cnt * self.buffer_size:
                                             (self.buffer_cnt + 1) * self.buffer_size + 1:],
                            self.channels[1][self.buffer_cnt * self.buffer_size:
                                             (self.buffer_cnt + 1) * self.buffer_size + 1:]]
            tmp_channels = np.array(tmp_channels)
            tmp_channels = np.ascontiguousarray(tmp_channels.T)
            tmp_sound = pygame.sndarray.make_sound(tmp_channels)

            while pygame.mixer.get_busy():
                sleep(0.01)

            sound = tmp_sound
            if not self.music_is_playing:
                return
            pygame.mixer.Sound.play(sound)

        tmp_channels = [self.channels[0][self.buffer_cnt * self.buffer_size::],
                        self.channels[1][self.buffer_cnt * self.buffer_size::]]
        tmp_channels = np.array(tmp_channels)
        tmp_channels = np.ascontiguousarray(tmp_channels.T)
        tmp_sound = pygame.sndarray.make_sound(tmp_channels)

        while pygame.mixer.get_busy():
            sleep(0.01)

        sound = tmp_sound
        if not self.music_is_playing:
            return
        pygame.mixer.Sound.play(sound)

        self.buffer_cnt = 0
        self.music_is_playing = False

    def music_edit(self, pos, value):
        old_value = self.sliders_old_values[pos]
        self.sliders_old_values[pos] = value

        if old_value == value:
            return

        if pos == 0:
            for i in range(self.nchannels):
                self.spectrum[i][:self.elem_per_hertz * self.bands[1][pos] + 1] *= 10 ** ((value - old_value) / 20)

        elif pos == 5:
            for i in range(self.nchannels):
                self.spectrum[i][self.elem_per_hertz * self.bands[0][pos]:] *= 10 ** ((value - old_value) / 20)

        else:
            for i in range(self.nchannels):
                self.spectrum[i][self.elem_per_hertz * self.bands[0][pos]:
                                 self.elem_per_hertz * self.bands[1][pos] + 1] *= 10 ** ((value - old_value) / 20)

        self.channels = (np.fft.irfft(self.spectrum)).astype(IntTypes.types[self.sampwidth])

        draw_1 = Worker(self.draw_array, self.spectrum, 0)
        self.threadpool.start(draw_1)

        draw_2 = Worker(self.draw_array, self.channels, 1)
        self.threadpool.start(draw_2)

    def redraw_subplot(self, canvas: dict, left, right=None):
        # потоконебезопасная функция, нужен mutex
        self.redraw_mutex.acquire()
        canvas['figure'].clear()
        subplot = canvas['figure'].add_subplot(1, 1, 1)
        subplot.set_xlabel(canvas['axes_labels'][0])
        subplot.set_ylabel(canvas['axes_labels'][1])
        canvas['figure'].align_xlabels()
        canvas['figure'].align_ylabels()
        # try:
        if right is not None:
            subplot.plot(left, right[:left.shape[0]])
        else:
            subplot.plot(left)
        # except ValueError:
        #     pass
        canvas['canvas'].draw()
        self.redraw_mutex.release()

    def draw_array(self, arr, spectrum_or_channel):
        if spectrum_or_channel == 0:
            self.redraw_subplot(self.canvases[1],
                                np.fft.rfftfreq(self.nframes, 1. / self.framerate)[::self.coefficient],
                                np.abs(arr[0][::self.coefficient]) / self.nframes)
        else:
            self.redraw_subplot(self.canvases[0], arr[0][::self.coefficient])

    def doing_vibrato(self, channels):
        fx = (AudioEffectsChain().pitch(shift=10))
        fx(self.path_to_pull, 'vibrato.wav')
        wav = wave.open('vibrato.wav', mode='r')
        (nchannels, sampwidth,
         _, nframes,
         _, _) = wav.getparams()
        content = wav.readframes(nframes)
        samples = np.fromstring(content, dtype=IntTypes.types[sampwidth])
        for i in range(nchannels):
            self.channels_vibrato.append(samples[i::nchannels])
        self.spectrum_vibrato = np.fft.rfft(self.channels_vibrato)

    def doing_reverberation(self, channels):
        fx = (AudioEffectsChain().reverb())
        fx(self.path_to_pull, 'reverberation.wav')
        wav = wave.open('reverberation.wav', mode='r')
        (nchannels, sampwidth,
         _, nframes,
         _, _) = wav.getparams()
        content = wav.readframes(nframes)
        samples = np.fromstring(content, dtype=IntTypes.types[sampwidth])
        for i in range(nchannels):
            self.channels_reverberation.append(samples[i::nchannels])
        self.spectrum_reverberation = np.fft.rfft(self.channels_reverberation)
