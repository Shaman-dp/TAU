import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

'''
start = -10
stop = 10
step = 0.01
y = np.arange(start, stop, step)
x = y

#Пробные сигналы

sig_sin = np.sin(y)
plt.plot(x, sig_sin)
plt.show()

sig_meandr = signal.square(y)
plt.plot(x, sig_meandr)
plt.show()

sig_saw = signal.sawtooth(y)
plt.plot(x, sig_saw)
plt.show()

#Типовые нелинейные звенья

sig_XXX_after_relay = np.sign(sig_XXX)
plt.plot(x, sig_XXX_after_relay)
plt.show()

def dead_zone_scalar(x, width = 0.5):
    if np.abs(x) < width:
        return 0
    elif x > 0:
        return x - width
    else:
        return x + width   
dead_zone = np.vectorize(dead_zone_scalar, otypes=[np.float])
sig_XXX_after_dead_zone = dead_zone(sig_XXX)
plt.plot(x, sig_XXX_after_dead_zone)
plt.show()

#понять как работает
def saturation_scalar(x, hight = 0.5):
    if np.abs(x) < hight:
        return x
    elif x > 0:
        return hight
    else:
        return -hight   
saturation = np.vectorize(saturation_scalar, otypes=[np.float])
sig_XXX_after_saturation = saturation(sig_XXX)
plt.plot(x, sig_XXX_after_saturation)
plt.show()

#Фильтрация

#понять как работает
k = 1
T = 1
B = [ k / (1 + T / 0.01) ]
A = [1, -1 / (1 + 0.01 / T)]
filtered_sig_XXX_after_YYY = signal.lfilter(B, A, np.sign(sig_XXX_after_YYY))
plt.plot(x, filtered_sig_XXX_after_YYY)
plt.show()

#Спектр

#Фурье
sig_spec = np.abs(fft(sig))
freqs = np.fft.fftfreq(<Размер вектора сигнала>, <Период дискретизации>)
plt.plot(freqs, sig_spec)
plt.show()
'''

#Амплитуды
A_sin = 1.2
A_meandr = 1.4
A_saw = 1.6 
#Частота
F = 1

#1
'''
start = -50
stop = 50
step = 1
y = np.arange(start, stop, step)
x = y

sig_sin = A_sin * np.sin(y)
plt.plot(x, sig_sin)
plt.xlabel('Время, сек.')
plt.ylabel('Амплитуда сигнала')
plt.title('Синусоида')
plt.grid(True)
plt.show()

sig_meandr = A_meandr * signal.square(y)
plt.plot(x, sig_meandr)
plt.xlabel('Время, сек.')
plt.ylabel('Амплитуда сигнала')
plt.title('Меандр')
plt.grid(True)
plt.show()

sig_saw = A_saw * signal.sawtooth(y)
plt.plot(x, sig_saw)
plt.xlabel('Время, сек.')
plt.ylabel('Амплитуда сигнала')
plt.title('Пилообразный сигнал')
plt.grid(True)
plt.show()
'''
#2
'''
sig_spec_sin = np.abs(np.fft.fft(sig_sin))
freqs = np.fft.fftfreq(sig_sin.size, F/2)
plt.plot(freqs, sig_spec_sin)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Спектр синусоидального сигнала')
plt.grid(True)
plt.show()

sig_spec_meandr = np.abs(np.fft.fft(sig_meandr))
freqs = np.fft.fftfreq(sig_meandr.size, F/2)
plt.plot(freqs, sig_spec_meandr)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Спектр меандра')
plt.grid(True)
plt.show()

sig_spec_saw = np.abs(np.fft.fft(sig_saw))
freqs = np.fft.fftfreq(sig_saw.size, F/2)
plt.plot(freqs, sig_spec_saw)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Спектр пилообразного сигнала')
plt.grid(True)
plt.show()
'''
#3, 4
'''
sig_sin_after_relay = np.sign(sig_sin)
plt.plot(x, sig_sin_after_relay)
plt.title('Идеальное реле для синусоидального сигнала')
plt.grid(True)
plt.show()

def dead_zone_scalar(x, width = 0.5):
    if np.abs(x) < width:
        return 0
    elif x > 0:
        return x - width
    else:
        return x + width   
dead_zone = np.vectorize(dead_zone_scalar, otypes=[np.float])
sig_sin_after_dead_zone = dead_zone(sig_sin)
plt.plot(x, sig_sin_after_dead_zone)
plt.title('Мертвая зона для синусоидального сигнала')
plt.grid(True)
plt.show()

def saturation_scalar(x, hight = 0.5):
    if np.abs(x) < hight:
        return x
    elif x > 0:
        return hight
    else:
        return -hight   
saturation = np.vectorize(saturation_scalar, otypes=[np.float])
sig_sin_after_saturation = saturation(sig_sin)
plt.plot(x, sig_sin_after_saturation)
plt.title('Усилитель с насыщением для синусоидального сигнала')
plt.grid(True)
plt.show()
#------------------------------------------------------------------#
sig_meandr_after_relay = np.sign(sig_meandr)
plt.plot(x, sig_meandr_after_relay)
plt.title('Идеальное реле для меандра')
plt.grid(True)
plt.show()

def dead_zone_scalar(x, width = 0.5):
    if np.abs(x) < width:
        return 0
    elif x > 0:
        return x - width
    else:
        return x + width   
dead_zone = np.vectorize(dead_zone_scalar, otypes=[np.float])
sig_meandr_after_dead_zone = dead_zone(sig_meandr)
plt.plot(x, sig_meandr_after_dead_zone)
plt.title('Мертвая зона для меандра')
plt.grid(True)
plt.show()

def saturation_scalar(x, hight = 0.5):
    if np.abs(x) < hight:
        return x
    elif x > 0:
        return hight
    else:
        return -hight   
saturation = np.vectorize(saturation_scalar, otypes=[np.float])
sig_meandr_after_saturation = saturation(sig_meandr)
plt.plot(x, sig_meandr_after_saturation)
plt.title('Усилитель с насыщением для меандра')
plt.grid(True)
plt.show()
#------------------------------------------------------------------#
sig_saw_after_relay = np.sign(sig_saw)
plt.plot(x, sig_saw_after_relay)
plt.title('Идеальное реле для пилообразного сигнала')
plt.grid(True)
plt.show()

def dead_zone_scalar(x, width = 0.5):
    if np.abs(x) < width:
        return 0
    elif x > 0:
        return x - width
    else:
        return x + width   
dead_zone = np.vectorize(dead_zone_scalar, otypes=[np.float])
sig_saw_after_dead_zone = dead_zone(sig_saw)
plt.plot(x, sig_saw_after_dead_zone)
plt.title('Мертвая зона для пилообразного сигнала')
plt.grid(True)
plt.show()

def saturation_scalar(x, hight = 0.5):
    if np.abs(x) < hight:
        return x
    elif x > 0:
        return hight
    else:
        return -hight   
saturation = np.vectorize(saturation_scalar, otypes=[np.float])
sig_saw_after_saturation = saturation(sig_saw)
plt.plot(x, sig_saw_after_saturation)
plt.title('Усилитель с насыщением для пилообразного сигнала')
plt.grid(True)
plt.show()
'''
#5
'''
sig_spec_sin_after_relay = np.abs(np.fft.fft(sig_sin_after_relay))
freqs = np.fft.fftfreq(sig_sin_after_relay.size, F/2)
plt.plot(freqs, sig_spec_sin_after_relay)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Спектр синусоидального сигнала после идеального реле')
plt.grid(True)
plt.show()

sig_spec_sin_after_dead_zone = np.abs(np.fft.fft(sig_sin_after_dead_zone))
freqs = np.fft.fftfreq(sig_sin_after_dead_zone.size, F/2)
plt.plot(freqs, sig_spec_sin_after_dead_zone)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Спектр синусоидального сигнала после мертвой зоны')
plt.grid(True)
plt.show()

sig_spec_sin_after_saturation = np.abs(np.fft.fft(sig_sin_after_saturation))
freqs = np.fft.fftfreq(sig_sin_after_saturation.size, F/2)
plt.plot(freqs, sig_spec_sin_after_saturation)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Спектр синусоидального сигнала после усилителя с насыщением')
plt.grid(True)
plt.show()
#------------------------------------------------------------------#
sig_spec_meandr_after_relay = np.abs(np.fft.fft(sig_meandr_after_relay))
freqs = np.fft.fftfreq(sig_meandr_after_relay.size, F/2)
plt.plot(freqs, sig_spec_meandr_after_relay)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Спектр меандра после идеального реле')
plt.grid(True)
plt.show()

sig_spec_meandr_after_dead_zone = np.abs(np.fft.fft(sig_meandr_after_dead_zone))
freqs = np.fft.fftfreq(sig_meandr_after_dead_zone.size, F/2)
plt.plot(freqs, sig_spec_meandr_after_dead_zone)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Спектр меандра после мертвой зоны')
plt.grid(True)
plt.show()

sig_spec_meandr_after_saturation = np.abs(np.fft.fft(sig_meandr_after_saturation))
freqs = np.fft.fftfreq(sig_meandr_after_saturation.size, F/2)
plt.plot(freqs, sig_spec_meandr_after_saturation)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Спектр меандра после усилителя с насыщением')
plt.grid(True)
plt.show()
#------------------------------------------------------------------#
sig_spec_saw_after_relay = np.abs(np.fft.fft(sig_saw_after_relay))
freqs = np.fft.fftfreq(sig_saw_after_relay.size, F/2)
plt.plot(freqs, sig_spec_saw_after_relay)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Спектр пилообразного сигнала после идеального реле')
plt.grid(True)
plt.show()

sig_spec_saw_after_dead_zone = np.abs(np.fft.fft(sig_saw_after_dead_zone))
freqs = np.fft.fftfreq(sig_saw_after_dead_zone.size, F/2)
plt.plot(freqs, sig_spec_saw_after_dead_zone)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Спектр пилообразного сигнала после мертвой зоны')
plt.grid(True)
plt.show()

sig_spec_saw_after_saturation = np.abs(np.fft.fft(sig_saw_after_saturation))
freqs = np.fft.fftfreq(sig_saw_after_saturation.size, F/2)
plt.plot(freqs, sig_spec_saw_after_saturation)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Спектр пилообразного сигнала после усилителя с насыщением')
plt.grid(True)
plt.show()
'''
#6
'''
k = 1
T = 1
B = [k / (1 + T/0.01)]
A = [1, -1/(1 + 0.01/T)]

filtered_sig_sin_after_relay = signal.lfilter(B, A, np.sign(sig_sin_after_relay))
plt.plot(x, filtered_sin_after_relay)
plt.title('Фильтр синусоидального сигнала после идеального реле')
plt.grid(True)
plt.show()

filtered_sig_sin_after_dead_zone = signal.lfilter(B, A, np.sign(sig_sin_after_dead_zone))
plt.plot(x, filtered_sin_after_dead_zone)
plt.title('Фильтр синусоидального сигнала после мертвой зоны')
plt.grid(True)
plt.show()

filtered_sig_sin_after_saturation = signal.lfilter(B, A, np.sign(sig_sin_after_saturation))
plt.plot(x, filtered_sig_sin_after_saturation)
plt.title('Фильтр синусоидального сигнала после усилителя с насыщением')
plt.grid(True)
plt.show()
#------------------------------------------------------------------#
filtered_sig_meandr_after_relay = signal.lfilter(B, A, np.sign(sig_meandr_after_relay))
plt.plot(x, filtered_meandr_after_relay)
plt.title('Фильтр меандра после идеального реле')
plt.grid(True)
plt.show()

filtered_sig_meandr_after_dead_zone = signal.lfilter(B, A, np.sign(sig_meandr_after_dead_zone))
plt.plot(x, filtered_meandr_after_dead_zone)
plt.title('Фильтр меандра после мертвой зоны')
plt.grid(True)
plt.show()

filtered_sig_meandr_after_saturation = signal.lfilter(B, A, np.sign(sig_meandr_after_saturation))
plt.plot(x, filtered_sig_meandr_after_saturation)
plt.title('Фильтр меандра после усилителя с насыщением')
plt.grid(True)
plt.show()
#------------------------------------------------------------------#
filtered_sig_saw_after_relay = signal.lfilter(B, A, np.sign(sig_saw_after_relay))
plt.plot(x, filtered_saw_after_relay)
plt.title('Фильтр пилообразного сигнала после идеального реле')
plt.grid(True)
plt.show()

filtered_sig_saw_after_dead_zone = signal.lfilter(B, A, np.sign(sig_saw_after_dead_zone))
plt.plot(x, filtered_saw_after_dead_zone)
plt.title('Фильтр пилообразного сигнала после мертвой зоны')
plt.grid(True)
plt.show()

filtered_sig_saw_after_saturation = signal.lfilter(B, A, np.sign(sig_saw_after_saturation))
plt.plot(x, filtered_sig_saw_after_saturation)
plt.title('Фильтр пилообразного сигнала после усилителя с насыщением')
plt.grid(True)
plt.show()
'''
#7
'''
sig_spec_filtered_sig_sin_after_relay = np.abs(np.fft.fft(filtered_sig_sin_after_relay))
freqs = np.fft.fftfreq(filtered_sig_sin_after_relay.size, F/2)
plt.plot(freqs, sig_spec_filtered_sig_sin_after_relay)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Фильтр спектра синусоидального сигнала после идеального реле')
plt.grid(True)
plt.show()

sig_spec_filtered_sig_sin_after_dead_zone = np.abs(np.fft.fft(filtered_sig_sin_after_dead_zone))
freqs = np.fft.fftfreq(filtered_sig_sin_after_dead_zone.size, F/2)
plt.plot(freqs, sig_spec_filtered_sig_sin_after_dead_zone)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Фильтр спектра синусоидального сигнала после мертвой зоны')
plt.grid(True)
plt.show()

sig_spec_filtered_sig_sin_after_saturation = np.abs(np.fft.fft(filtered_sig_sin_after_saturation))
freqs = np.fft.fftfreq(filtered_sig_sin_after_saturation.size, F/2)
plt.plot(freqs, sig_spec_filtered_sig_sin_after_saturation)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Фильтр спектра синусоидального сигнала после усилителя с насыщением')
plt.grid(True)
plt.show()
#------------------------------------------------------------------#
sig_spec_filtered_sig_meandr_after_relay = np.abs(np.fft.fft(filtered_sig_meandr_after_relay))
freqs = np.fft.fftfreq(filtered_sig_meandr_after_relay.size, F/2)
plt.plot(freqs, sig_spec_filtered_sig_meandr_after_relay)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Фильтр спектра меандра после идеального реле')
plt.grid(True)
plt.show()

sig_spec_filtered_sig_meandr_after_dead_zone = np.abs(np.fft.fft(filtered_sig_meandr_after_dead_zone))
freqs = np.fft.fftfreq(filtered_sig_meandr_after_dead_zone.size, F/2)
plt.plot(freqs, sig_spec_filtered_sig_meandr_after_dead_zone)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Фильтр спектра меандра после мертвой зоны')
plt.grid(True)
plt.show()

sig_spec_filtered_sig_meandr_after_saturation = np.abs(np.fft.fft(filtered_sig_meandr_after_saturation))
freqs = np.fft.fftfreq(filtered_sig_meandr_after_saturation.size, F/2)
plt.plot(freqs, sig_spec_filtered_sig_meandr_after_saturation)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Фильтр спектра меандра после усилителя с насыщением')
plt.grid(True)
plt.show()
#------------------------------------------------------------------#
sig_spec_filtered_sig_saw_after_relay = np.abs(np.fft.fft(filtered_sig_saw_after_relay))
freqs = np.fft.fftfreq(filtered_sig_saw_after_relay.size, F/2)
plt.plot(freqs, sig_spec_filtered_sig_saw_after_relay)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Фильтр спектра пилообразного сигнала после идеального реле')
plt.grid(True)
plt.show()

sig_spec_filtered_sig_saw_after_dead_zone = np.abs(np.fft.fft(filtered_sig_saw_after_dead_zone))
freqs = np.fft.fftfreq(filtered_sig_saw_after_dead_zone.size, F/2)
plt.plot(freqs, sig_spec_filtered_sig_saw_after_dead_zone)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Фильтр спектра пилообразного сигнала после мертвой зоны')
plt.grid(True)
plt.show()

sig_spec_filtered_sig_saw_after_saturation = np.abs(np.fft.fft(filtered_sig_saw_after_saturation))
freqs = np.fft.fftfreq(filtered_sig_saw_after_saturation.size, F/2)
plt.plot(freqs, sig_spec_filtered_sig_saw_after_saturation)
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда сигнала')
plt.title('Фильтр спектра пилообразного сигнала после усилителя с насыщением')
plt.grid(True)
plt.show()
'''