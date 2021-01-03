# AssignmentA

<table>
<tr>
<td>
  A program for solving simple handwritten mathematical operations by taking a picture of the equation. It uses a convoultional neural network trained on handwritten mathematical symbols.
  The model is trained on the HASYv2 database (https://www.kaggle.com/guru001/hasyv2)
  The output is a list of strings that get parsed and solved showing steps in case the problem has brackets.
</td>
</tr>
</table>


## Instruction
* _final.py__ is the main file
  * It requires image path (currently holds 'testing.png')
  * Running the file will print the equation from the image and a solution
    In case there are brackets in the equation it will print each step
* _image_pre.py_ preprocesses the database and builds and saves the model
  * It required a dataset folder path (database has to be downloaded)
  * This file isn't crucial in running the model as the repository already contains the model folder and database classes file
