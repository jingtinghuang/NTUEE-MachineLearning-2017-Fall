from keras.utils.vis_utils import plot_model
from keras.models import load_model

def main():
    

    emotion_classifier = load_model("./81816/LSTM_semi3.h5")
    emotion_classifier.summary()
    plot_model(emotion_classifier,to_file='./rnn.png')

    emotion_classifier = load_model("./LSTM.h5")
    emotion_classifier.summary()
    plot_model(emotion_classifier,to_file='./bow.png')

if __name__=='__main__':
    main()