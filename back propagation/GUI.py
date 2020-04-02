from tkinter import  *
from tkinter import  messagebox
import numpy as np
import helper as hlp
# from Plot_Features_Combinations import line
from BackPropragation import backPropragation as model

def Run(num_layers,num_neurons,epochs,learningRate,add_bias,activation):
    try:
        num_layers = int(num_layers)
        epochs = int(epochs)
        learningRate = float(learningRate)
    except ValueError as e:
        messagebox.showerror(message="please Enter only numerical values.")
        return

    try:
        num_neurons=list(map(int,num_neurons.split(',')))
        if len(num_neurons)!=num_layers:
            raise ValueError("Wrong number of layers")
    except ValueError as e:
        messagebox.showerror(message="please Enter {} numbers for each layer separated by `,` ".format(num_layers))
        return

    xy = hlp.set_Data(model.no_of_features,model.no_of_classes, add_bias)
    print("done")
    activation=model.Sigmoid if activation=="Sigmoid" else model.tanh
    m=model(num_layers,num_neurons,epochs,learningRate,add_bias,xy,activation)
    y_pred=m.predict()
    acc=np.mean(y_pred==m.y_test)
    print(acc)
Run(2,"2,5",100,0.1,0,"Sigmoid")

def buildGui():
    root = Tk()
    root.title('Task 1')
    root.geometry("300x300")
    OptionList = [
        "Sigmoid",
        "Hyperbolic Tangent sigmoid"
    ]

    hidden_layers_label = Label(root, text='Enter #hidden Layers:').pack()
    hidden_layers_entry=Entry(root)
    hidden_layers_entry.pack()

    hidden_neurons_label = Label(root, text='Enter #neurons in each Layer:').pack()
    hidden_neurons_entry = Entry(root)
    hidden_neurons_entry.pack()

    learning_rate_label = Label(root, text='Enter Learning Rate:').pack()
    learning_rate_entry=Entry(root)
    learning_rate_entry.pack()

    epochs_label=Label(root,text='Enter #Epochs:').pack()
    epochs_entry=Entry(root)
    epochs_entry.pack()
    bias_var=IntVar()
    bias_checkButton=Checkbutton(root,text="Add Bias",variable=bias_var).pack()

    activation = StringVar(root)
    activation.set(OptionList[0])
    opt = OptionMenu(root, activation, *OptionList)
    opt.pack()

    run_button=Button(text ="Run",command = lambda :
    Run(
        hidden_layers_entry.get(),
        hidden_neurons_entry.get(),
        learning_rate_entry.get(),
        epochs_entry.get(),
        bias_var.get(),
        activation.get()
    )).pack()
    root.mainloop()



#buildGui()
