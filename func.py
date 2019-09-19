#def abc(s):
#        nw = s** 2
#        print(nw)
#abc(12)

#def square(value):
#        newVal = value ** 2
#        return newVal
#num = square(3)
#print(num)

#def raiseToPower( v1, v2, v3 ):
#        """raise to power"""
#        newV1 = (v1 * v2) + v3
#        newV2 = v1 + v3
#        newTuple = (newV1, newV2)
#        return newTuple
#
#s = raiseToPower(1, 2, 3)
#print(s)

#def maxi( v1, v2 ):
#        if v1 > v2:
#                return v1
#        else:
#                return v2
#
#print( maxi( 2, 5))

#def overlapping( lst1, lst2 ):
#        for i in lst1:
#                for j in lst2:
#                        if i == j:
#                                return True
#        return False
#
#print( overlapping( ['pthis', 'might', 'pthis'], ['or', 'maybe', 'this'] ) )
#        

#def generateNchars( n, pr):
#        result = ""
#        for x in range(n):
#                result += pr
#        return result
#print( generateNchars( 5, "x" ) )

#def histogram( lst ):
#        for n in lst:
#                print( n * " * " )
#                print( 10 * " - " )
#histogram( [4, 9, 3] )

#def wordsLength( lst ):
#        lenlist = [ len( i ) for i in lst ]
#        return lenlist
#print( wordsLength( [ 'Hello', 'World!'] ) )

#def sqrt( x ):
#        try:
#                return x ** 0.5
#        except:
#                print('x must be int')
#s = sqrt( 'f' )
#print( s )

def filterLongWords( lst, n ):
        return [ w for w in lst if len( w ) > n ]
print( filterLongWords( ['asdfs', 'aesfdsdg', 'asd'], 5 ) )


