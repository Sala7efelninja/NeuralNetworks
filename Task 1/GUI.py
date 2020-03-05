from tkinter import  *
from tkinter import  messagebox


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

root.mainloop()