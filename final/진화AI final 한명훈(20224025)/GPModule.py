import numpy as np
from numpy import dot
from numpy.linalg import norm
from GP import *
from GA import *
import copy
import math

class GPModule:
    max_depth = 15
    module_ganerator_list = [
        lambda input_num, depth : GPConstant(input_num, depth),
        lambda input_num, depth : GPVariable(input_num, depth),
        lambda input_num, depth : GPOperator(input_num, depth),
    ]
    def __init__(self, input_num, depth) -> None:
        self.input_num = input_num # 입력 인자 개수 (입력 리스트의 크기)
        self.depth = depth

    def calculate(self, input) -> float:
        pass

    @staticmethod
    def get_new_module(input_num, depth): # depth에 위치할 모듈을 생성하여 반환
        if GPModule.max_depth == depth:
            # 값(상수, 변수)에 해당하는 모듈 중 하나를 생성하여 반환
            return GPModule.module_ganerator_list[random.randint(0, 1)](input_num, depth)
        else:
            # 모든 유형의 모듈 중 하나를 생성하여 반환
            probability = random.random()
            if probability <= 0.1:
                idx = 0
            elif probability <= 0.2:
                idx = 1
            else:
                idx = 2
            return GPModule.module_ganerator_list[idx](input_num, depth)

    @staticmethod
    def change_random_sub_tree(tree1, tree2) -> None:
        node1 = tree1.get_random_replacable_module()
        node2 = tree2.get_random_replacable_module()

        if (node1 == None) or (node2 == None):
            return

        idx1 = random.randint(0, node1.get_children_num()-1)
        idx2 = random.randint(0, node2.get_children_num()-1)

        child1 = node1.get_child(idx1)
        child2 = node2.get_child(idx2)

        node1.set_child(idx1, child2)
        node2.set_child(idx2, child1)

    @staticmethod
    def mutation(tree) -> None:
        # 임의의 서브 노드 바꾸기
        module_list = tree.get_all_replacable_modules()

        if 1 < len(module_list):
            idx_list = []
            for i, module in enumerate(module_list):
                idx_list.extend([i for j in range(module.depth//5+1)])
            module = module_list[idx_list[random.randint(0, len(idx_list)-1)]]

            idx = random.randint(0, module.get_children_num()-1)


            child = GPModule.get_new_module(module.input_num, module.depth+1)
            module.set_child(idx, child)
            return True

        # 실패시 False 반환
        return False

    @staticmethod
    def change_constant_value(tree):
        module = tree.get_random_constant_module()
        if module == None:
            return False
        module.value = random.randint(-3, 3)
        return True
        
    def size(self) -> int:
        return 1

    def copy(self):
        return copy.deepcopy(self)

    def get_all_replacable_modules(self) -> list:
        # 자신의 자식 모듈을 바꿀 수 있는 모듈 모두 반환
        pass

    def get_random_replacable_module(self):
        replacable_modules = self.get_all_replacable_modules()
        if len(replacable_modules) == 0:
            return None
        return replacable_modules[random.randint(0, len(replacable_modules)-1)]

    def get_random_constant_module(self):
        constant_modules = self.get_constant_modules()
        if len(constant_modules) == 0:
            return None
        return constant_modules[random.randint(0, len(constant_modules)-1)]

    def get_constant_modules(self):
        # 자기 자신과 자식들 중 상수 모듈을 모두 반환
        pass


    def get_children_num(self) -> int:
        pass

    def get_child(self, idx):
        pass

    def set_child(self, idx, child) -> None:
        pass

    def to_string() -> str:
        pass

class OperatorID(IntEnum):
    PLUS = 0
    MINUS = 1
    MULITPLY = 2
    DIVIDE = 3
    SIN = 4
    COS = 5
    POWER = 6
    ROOT = 7
    RELU = 8
    ABS = 9
    ABS_LOG = 10

operator_input_num = {
    OperatorID.PLUS:2,
    OperatorID.MINUS:2,
    OperatorID.MULITPLY:2,
    OperatorID.DIVIDE:2,
    OperatorID.SIN:1,
    OperatorID.COS:1,
    OperatorID.POWER:1,
    OperatorID.ROOT:1,
    OperatorID.RELU:1,
    OperatorID.ABS:1,
    OperatorID.ABS_LOG:1,
}

class GPOperator(GPModule):
    
    def __init__(self, input_num, depth) -> None:
        super().__init__(input_num, depth)
        self.opratorID = random.randint(0, len(OperatorID) -1) 
        self.children = []
        for i in range(self.get_input_num()): # 자식 채우기
            self.children.append(GPModule.get_new_module(self.input_num, depth+1))
    
    def calculate(self, input) -> float:
        values = [self.children[i].calculate(input) for i in range(len(self.children))]
        if self.opratorID == OperatorID.PLUS:
            return values[0] + values[1]
        elif self.opratorID == OperatorID.MINUS:
            return values[0] - values[1]
        elif self.opratorID == OperatorID.MULITPLY:
            return values[0] * values[1]
        elif self.opratorID == OperatorID.DIVIDE:            
            if values[1] == 0:
                return 0
            return values[0] / values[1]
        elif self.opratorID == OperatorID.SIN:
            return math.sin(values[0])
        elif self.opratorID == OperatorID.COS:
            return math.cos(values[0])
        elif self.opratorID == OperatorID.POWER:
            return values[0]**2
        elif self.opratorID == OperatorID.ROOT:
            return max(0, values[0])**(1/2)
        elif self.opratorID == OperatorID.RELU:
            return max(values[0], 0)
        elif self.opratorID == OperatorID.ABS:
            return abs(values[0])
        elif self.opratorID == OperatorID.ABS_LOG:
            return math.log(max(0.00001, abs(values[0])))

    def get_input_num(self) -> int:
        return operator_input_num[self.opratorID]

    def size(self) -> int:
        sum = 0
        for child in self.children:
            sum += child.size()
        return sum + 1

    def mutation(self) -> bool:
        # 자식 노드 중 하나를 새로 생성한 모듈로 대체
        # 자식 모듈 중 하나에 대해 mutation()을 수행하고 실패시
        # 일정 확률로 자식 모듈 변경 또는 실패 반환
        idx = random.randint(0, len(self.children)-1)
        result = self.children[idx].mutation()
        if result == False:
            if random.random() <= 0.1:
                self.children[idx] = GPModule.get_new_module(self.input_num, self.depth+1)
                return True
            else:
                return False
        else:
            return True
    
    def print_state(self) -> None:
        # generation 및 fitness 정보 출력
        pass

    def get_all_replacable_modules(self) -> list:
        modules = [self]
        for child in self.children:
            modules.extend(child.get_all_replacable_modules())

        return modules

    def get_children_num(self) -> int:
        return len(self.children)

    def get_child(self, idx) -> GPModule:
        return self.children[idx]
        
    def set_child(self, idx, child) -> None:
        self.children[idx] = child

    def to_string(self) -> str:
        values = [child.to_string() for child in self.children]
        if self.opratorID == OperatorID.PLUS:
            return "(%s + %s)"%(values[0], values[1])
        elif self.opratorID == OperatorID.MINUS:
            return "(%s - %s)"%(values[0], values[1])
        elif self.opratorID == OperatorID.MULITPLY:
            return "(%s * %s)"%(values[0], values[1])
        elif self.opratorID == OperatorID.DIVIDE:
            return "(%s / %s)"%(values[0], values[1])
        elif self.opratorID == OperatorID.SIN:
            return "sin(%s)"%(values[0])
        elif self.opratorID == OperatorID.COS:
            return "cos(%s)"%(values[0])
        elif self.opratorID == OperatorID.POWER:
            return "(%s**2)"%(values[0])
        elif self.opratorID == OperatorID.ROOT:
            return "(max(0, %s)**(1/2))"%(values[0])
        elif self.opratorID == OperatorID.RELU:
            return "max(0, %s)"%(values[0])
        elif self.opratorID == OperatorID.ABS:
            return "abs(%s)"%(values[0])
        elif self.opratorID == OperatorID.ABS_LOG:
            return "log(abs(%s))"%(values[0])

    def get_constant_modules(self) -> list:
        # 자기 자신과 자식들 중 상수 모듈을 모두 반환
        modules = []
        for child in self.children:
            modules.extend(child.get_constant_modules())
        return modules


class GPVariable(GPModule):
    def __init__(self, input_num, depth) -> None:
        super().__init__(input_num, depth)
        self.variable_idx = random.randint(0, self.input_num-1) # 입력 값 중 랜덤으로 하나 지정

    def calculate(self, input) -> float:
        return input[self.variable_idx] # 선택된 입력 값 반환

    def mutation(self) -> bool:
        return False

    def get_all_replacable_modules(self) -> list:
        return []

    def get_children_num(self) -> int:
        return 0

    def to_string(self) -> str:
        return ["x", "y", "z"][self.variable_idx]

    def get_constant_modules(self) -> list:
        # 자기 자신과 자식들 중 상수 모듈을 모두 반환
        return []

class GPConstant(GPModule):
    # (min, max) 사이의 상수값을 나타낸다. 
    min = -3
    max = 3
    def __init__(self, input_num, depth) -> None:
        super().__init__(input_num, depth)
        self.value = random.uniform(GPConstant.min, GPConstant.max) # 범위 내에서 랜덤으로 상수 값 지정

    def calculate(self, input) -> float:
        return self.value # 지정된 상수 값 반환

    def mutation(self) -> bool:
        return False
    
    def get_all_replacable_modules(self) -> list:
        return []

    def get_children_num(self) -> int:
        return 0

    def to_string(self) -> str:
        return "%f"%self.value

    def get_constant_modules(self) -> list:
        # 자기 자신과 자식들 중 상수 모듈을 모두 반환
        return [self]