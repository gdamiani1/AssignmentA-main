"""
Solver and parser for Assignment A
"""


def string_separator(equation):
    """
    Converts the list of strings into a list of integers where operators are represented as a negative integer,
    and numbers as positive integers.
    It also connects adjacent numbers into a single two or more digit number

    Input:      List of strings in which each string represents a prediction made by the model
    Output:     List of integers explained above
    """
    eltype = []
    intcheck = False
    for elem in equation:
        if elem == '0' or elem == '1' or elem == '2' or elem == '3' \
                or elem == '4' or elem == '5' or elem == '6' or elem == '7' or elem == '8' or elem == '9':
            if intcheck:
                eltype[-1] = eltype[-1] * 10 + int(elem)
            else:
                eltype.append(int(elem))
                intcheck = True
        elif elem == '+' or elem == '\\pm':
            eltype.append(-1)
            intcheck = False
        elif elem == '-' or elem == '\\neg':
            eltype.append(-2)
            intcheck = False
        elif elem == '/':
            eltype.append(-3)
            intcheck = False
        elif elem == '\\times':
            eltype.append(-4)
            intcheck = False
        elif elem == '\\{' or elem == '\\iota' or elem == '\\langle' \
                or elem == '\\llbracket' or elem == '\\ell' or elem == '[':
            eltype.append(-5)
            intcheck = False
        elif elem == '\\gamma' or elem == '\\rangle' or elem == '\\}' \
                or elem == 'j' or elem == ']' or elem == '\\rrbracket':
            eltype.append(-6)
            intcheck = False

    return eltype


def brackets(eltype):
    """
    Checks if the eqation contains brackets

    Input:      Equation in the form of an integer list
    Output:     True if there are brackets in the equation
                False if there are no brackets in the equaiton
    """
    for e in range(0, len(eltype)):
        if eltype[e] == -5 or eltype[e] == -6:
            return True
    return False


def printeq(equation):
    """
    Prints the equation

    Input:      List of integers where every integer is represented with a positive integer,
                and an operation with a negative integer
    Output:     Prints the equation
    """
    string = []
    for e in equation:
        if e >= 0:
            string.append(str(e))
        if e == -1:
            string.append('+')
        if e == -2:
            string.append('-')
        if e == -3:
            string.append('/')
        if e == -4:
            string.append('*')
        if e == -5:
            string.append('(')
        if e == -6:
            string.append(')')
    print(''.join(map(str, string)))


def bracketSplit(equation):
    """
    Called when there is a bracket in the equation,
    it splits calculates the inside of the bracket and prints it

    Input:      The full equation    (Only if it contains brackets)
    Output:     The equation with the brackets calculated
    """
    leftb = 0
    rightb = len(equation) - 1
    for e in range(0, len(equation)):
        if equation[e] == -5:
            leftb = e
        if equation[e] == -6:
            rightb = e
            break
    inside = []
    for c in range(leftb + 1, rightb):
        inside.append(equation[c])

    result = calculate(inside)
    del equation[leftb:(rightb + 1)]
    equation.insert(leftb, result)
    printeq(equation)
    if brackets(equation):
        bracketSplit(equation)

    return equation


def calculate(eq):
    """
    Calculates the whole equation if it doesn't have brackets,
    otherwise, the input is the part of the equation inside of the bracket.

    Input:      List of integers representing elements of the equation
    Output:     Solution of the equation
    """
    if len(eq) == 1:
        return eq[0]
    elif len(eq) == 2:
        return eq[1] * -1
    for e in range(0, len(eq) - 2):
        if eq[e + 1] == -4:
            if eq[e + 2] == -2:
                eq[e] = eq[e] * (eq[e + 3] * -1)
                eq.pop(e + 1)
                eq.pop(e + 1)
                eq.pop(e + 1)

            else:
                eq[e] = eq[e] * eq[e + 2]
                eq.pop(e + 1)
                eq.pop(e + 1)
        elif eq[e + 1] == -3:
            if eq[e + 2] == -2:
                eq[e] = eq[e] / (eq[e + 3] * -1)
                eq.pop(e + 1)
                eq.pop(e + 1)
                eq.pop(e + 1)
            else:
                eq[e] = eq[e] / eq[e + 2]
                eq.pop(e + 1)
                eq.pop(e + 1)

    for e in range(0, len(eq) - 2):
        if eq[e + 1] == -1:
            if eq[e + 2] == -2:
                eq[e] = eq[e] + (eq[e + 3] * -1)
                eq.pop(e + 1)
                eq.pop(e + 1)
                eq.pop(e + 1)
            else:
                eq[e] = eq[e] + eq[e + 2]
                eq.pop(e + 1)
                eq.pop(e + 1)
        elif eq[e + 1] == -2:
            if eq[e + 2] == -2:
                eq[e] = eq[e] + eq[e + 3]
                eq.pop(e + 1)
                eq.pop(e + 1)
                eq.pop(e + 1)
            else:
                eq[e] = eq[e] - eq[e + 2]
                eq.pop(e + 1)
                eq.pop(e + 1)
    return eq[0]


def solve(str_eq):
    """
    Final function of the parser/solver.

    Input:      List of strings (model predictions)
    Output.     Prints the full equation
                Prints each step of solving (if the equation contains brackets)
                Prints the final result
    """
    final = string_separator(str_eq)
    printeq(final)
    if brackets(final):
        final = bracketSplit(final)
    print(calculate(final))
