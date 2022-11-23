
The algo responds to the famous knight's tour problem enunciated by Euler. 

A knight's tour is a sequence of moves of a knight on a chessboard such that the knight visits every square exactly once. If the knight ends on a square that is one knight's move from the beginning square (so that it could tour the board again immediately, following the same path), the tour is closed (or re-entrant); otherwise, it is open. 
It corresponds to finding a solution to a hamilonian graph problem in graph theory. 

This code shows a simple algorithm able to do this. 
On a 8*8 standard chess squared board, you can run the code for basically every initial position and the algorithm converges outputting the sequence of positions. 
You can test and vary the board size (given that you stay on a square board) (**by looking at smaller boards, you get blocked really fast by the limited size (the knight does not have enough space to move).

The technique used to solve the algorithm is one using "adjacent matrices" and their different power (k) meaning matrices computing the nb of paths of lenght k between two points.    
The algorithm that has proven sucessful is one that minimizes the number of potential paths. Please feel free to exchange on the  mathematical justification behind the convergence (which, unformally, is translated by "giving privilege to the cases that are hardest to get to"). 

![image](https://user-images.githubusercontent.com/74412016/203449933-d9b88ac9-3ee3-476d-9a7b-96dba8d7d069.png)
