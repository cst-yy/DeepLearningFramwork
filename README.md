# DeScaleFed
## A deep learning framework

***

###  conda：
>
+  python =3.9
+  numpy
+  matplotlib
+  Pillow
+  CuPy

### **Notes:**
1. Forward propagation uses numpy for calculations :)
2. To calculate the derivative of the derivative, please add when calling backpropagation: Config.enable_backprop
3. To update the parameters or in any occasion that does not require derivation, please manually call: see the test.py example for details
   ~~~python
    with no_grad():
        xxx
        ...
   ~~~
4. If the function of the Variable class adds @property, you can directly treat it as a variable without explicitly calling
5. The function module has added support for automatic broadcasting
6. For bugs that cannot be determined in the complex backpropagation process, it is recommended to use the '.dot' calculation graph for explicit analysis, and call the following function: 
    ~~~python
    utils.plt_dot_graph(loss, verbose=False, to_file='lossname.png')
    ~~~

***
###  To-do list 
- [X] Support for optimizers
- [X] Support for activation functions
- [X] Auto load data support
- [ ] GPU computing support
