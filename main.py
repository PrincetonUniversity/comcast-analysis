from analysis-simple import *

if __name__=='__main__':
    test = ComcastDataSet(CONST)
    control = ComcastDataSet(CONST)

    test.name = '250-test'
    control.name = 'control1'

    test.load()
    control.load()
