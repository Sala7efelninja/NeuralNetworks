from tkinter import  *
from tkinter import  messagebox
import numpy as np
import helper as hlp
import Perceptron as prc
from Plot_Features_Combinations import line


class Checkbar(Frame):
    def __init__(self, parent=None, picks=[],text='select Feature',row_Num=0,col_num=0, side=LEFT, anchor=W):
        Frame.__init__(self, parent)
        self.vars = []
        i=1
        l = Label(parent, text=text)
        l.grid(column=col_num,row=row_Num)
        for pick in picks:
            var = IntVar()
            chk = Checkbutton(self, text=pick, variable=var,command=self.check_states)
            chk.grid(column=col_num+i,row=row_Num+1)
            i=i+1
            self.vars.append(var)
    def state(self):
        return [var.get() for var in self.vars]

    def check_states(self):
       states=self.state()
       s=sum(states)

       if s>2:
           messagebox.showerror(message="You must check 2 boxes only")
           [var.set(0) for var in self.vars]


def get_selected_indeces(check_list):
    selected=[]
    for i in range(len(check_list)):
        if check_list[i]:
            selected.append(i)
    return selected

def Run_Perceptron(features,classes,epochs,learningRate,add_bias,thresh,stopping_condition="epochs"):
    if stopping_condition=="epochs":
       epochs=int(epochs)
    elif stopping_condition=="thresh":
        thresh=float(thresh)
    learningRate=float(learningRate)
    selected_features=get_selected_indeces(features)
    selected_classes=get_selected_indeces(classes)
    x_train, x_test, y_train, y_test = hlp.set_Data(selected_classes[0], selected_classes[1], selected_features[0],selected_features[1],add_bias)
    #W=prc.fit(x_train,y_train,epochs,learningRate)
    if stopping_condition=="epochs":
        W=prc.fit_adaline(x_train,y_train,epochs,learningRate)
    elif stopping_condition == "thresh":
        W = prc.fit_adaline(x_train, y_train, learningRate, thresh)
    print(W)
    y_pred=prc.predict(x_test,W)
    acc,cm=prc.evaluate_model(y_test,y_pred)
    print(acc)
    print(cm)
    line(y_test,x_test,W)




f=[1,1,0,0]
c=[1,1,0,0]
Run_Perceptron(f,c,0,0.01,True,0.5,stopping_condition="thresh")


def buildGui():
    root = Tk()
    root.title('Task 1')
    root.geometry("300x300")
    feature_checkBoxBar = Checkbar(root,row_Num=0,col_num=0,picks= ['X1', 'X2', 'X3', 'X4'],text='Select Feature :')
    feature_checkBoxBar.grid(column=0,row=1)
    class_checkBoxBar = Checkbar(root,row_Num=5,col_num=0,picks=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],text='Select Class :')
    class_checkBoxBar.grid(column=0,row=7)
    learning_rate_label=Label(root,text='Enter Learning Rate:')
    learning_rate_label.grid(column=0,row=8)
    learning_rate_entry=Entry(root)
    learning_rate_entry.grid(column=0,row=9)
    epochs_label=Label(root,text='Enter #Epochs:')
    epochs_label.grid(column=0,row=10)
    epochs_entry=Entry(root)
    epochs_entry.grid(column=0,row=11)
    bias_var=IntVar()
    bias_checkButton=Checkbutton(root,text="Add Bias",variable=bias_var)
    bias_checkButton.grid(column=0, row=12)
    thresh_label=Label(root,text='Enter threshold:')
    thresh_label.grid(column=0,row=13)
    thresh_entry=Entry(root)
    thresh_entry.grid(column=0,row=14)
    run_button=Button(text ="Run",command = lambda :
    Run_Perceptron(feature_checkBoxBar.state(),class_checkBoxBar.state(),epochs_entry.get(),learning_rate_entry.get(),bias_var.get(),thresh_entry.get(),stopping_condition="thresh"))
    run_button.grid(column=0,row=15)

    root.mainloop()





# buildGui()
