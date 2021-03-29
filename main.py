import test_accuracy_cnn_model
import pytorch_training_cnn_model
import image_prediction_using_model
from tkinter import filedialog
from tkinter import *

def main():
    test_accuracy_cnn_model.test_accuracy()

    window = Tk()
    window.resizable(False, False)
    window.geometry('350x125')
    window.title("GLaDOS")
    title_label = Label(window, text="Flower Prediction", font=("Arial Bold", 30))
    title_label.grid(column=0, row=0)
    prediction_label = Label(window, text="Prediction", font=("Arial Bold", 15))
    prediction_label.grid(column=0, row=3)

    def predict_button_clicked():
        prediction_label.configure(text="Processing...")
        window.filename = filedialog.askopenfilename(
            initialdir="D:/Facultate/AN III/Semester 2/Intelligent Systems/venv/flowers", title="Select file",
            filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        if not window.filename:
            prediction_label.configure(text="No file selected. :(")
            return

        path_to_file = window.filename[56:]
        flower_name = image_prediction_using_model.predict(path_to_file)
        prediction_label.configure(text="My prediction is: " + flower_name)

    predict_button = Button(window, text="Predict", command=predict_button_clicked)
    predict_button.grid(column=0, row=1)

    window.mainloop()


if __name__ == "__main__":
    main()
