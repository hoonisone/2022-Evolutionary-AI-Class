from GPFinal import *
from math import sin, cos



gp = GPFinal(100, 2, cross_over_p=0.1, mutation_p=1)

# while True:
#     individual = gp.get_best_individual()
#     loss = gp.loss(individual)
#     print(loss)
#     if loss <20000:
#         break
    
#     gp.population = gp.generate_population()


gp.print_state()

for i in range(250):
    gp.next_population1(1)
    gp.print_state()



print(gp.get_best_individual().to_string())

# x = -4.513431
# y = 0.382586
# x = 3.687974
# y = 3.725057
# z = ((((sin(cos(cos(0.124038))) + (cos(cos(((-2.323298**(1/2)) - -1.375740))) * 2.791166)) + (cos(((abs(((abs((-0.785789 / x)) / 0.888430)**(1/2))) + ((cos(sin((((cos(sin(cos(cos(-0.845875))))**2) / abs(((cos(sin(y)) - x)**2)))**(1/2))))**(1/2)) + sin((-2.920992**(1/2))))) + (((x / abs(y)) - ((2.179309 + 0.344109) / y)) * (abs((1.874398 - 0.842434)) - sin(((-0.015686**(1/2)) - max(0, sin(max(0, (((((sin((((sin(abs((((cos(x) * 1.058798) + ((-0.102764 / x)**(1/2)))**(1/2)))) + (1.760098 - sin((cos(((x - -0.032192) / (y * -2.982257)))**(1/2))))) + y) / cos(((((1.279572 * y)**2) - ((max(0, (-1.522633 - -2.088303))**(1/2)) + (-2.531216 * (cos(cos((1.127681 + x))) - (abs(abs(-2.525738))**(1/2)))))) * (max(0, (((sin(((sin(((((max(0, max(0, (sin((cos(y) * max(0, y)))**2))) - x) + abs(-0.764029)) + y) / cos(((((cos(((abs(cos(abs(x))) + (abs(-0.764029) + sin((-2.920992**(1/2))))) + ((((-2.440774 * y) + 0.444202) - max(0, max(0, x))) * -2.241346)))**2)**2) - ((max(0, (((sin((y / y)) + ((x**(1/2))**2))**(1/2))**2))**(1/2)) + (((cos(cos(1.544071))**2)**(1/2)) * (cos(cos((1.127681 + x))) - ((0.237317 / (2.131908 + y)) - (abs(y) - (y * -0.394475))))))) * x))))**(1/2))**(1/2))) + (cos(cos((y - (abs(0.039385) - (abs(-2.440497) / -2.802563))))) * 2.791166)) + (1.906812**2)) / (-1.579640 + y))) * ((abs(0.842434) / 0.888430)**(1/2)))))))**(1/2))**(1/2))**2)**2)**(1/2)))))))))))**2)) / cos((2.237950**2)))**2)
# z = (((((2.737679 - cos((((abs(1.654114) + (-2.047128**2)) + 1.190727) / y))) + ((max(0, sin(sin(cos((y**2)))))**(1/2))**2)) * (abs(max(0, (max(0, (((sin((max(0, (max(0, (sin((x**2)) * abs(x)))**(1/2)))**(1/2))) + ((max(0, sin(1.466094))**(1/2))**2)) * (abs(y) + 0.919493)) + 2.887642))**(1/2)))) + -0.461230)) + 2.887642)**2)
# print(z)