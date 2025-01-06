import re


class EquationExtractor:
    def __init__(self):
        self.variables = []
        self.values = []
        self.equations = {}

    def __clean_equation(self):
        self.variables.clear()
        self.values.clear()
        self.equations.clear()

    def __extract(self, text) -> dict:
        """
        从文本中提取方程等式
        :param text: str
        :return: dict
        """
        self.__clean_equation()
        # 正则表达式用于匹配形如 x = ... 的等式
        equation_pattern = r"(\b[a-zA-Z]\w*\s*=\s*[^=]+)"
        for equation in re.findall(equation_pattern, text):
            # 对等式进行进一步解析
            parts = equation.split("=")
            if len(parts) == 2:
                key, value = parts
                # 去除空白并赋值
                key = key.strip()
                value = value.strip()
                # 移除值两侧的引号，如果有的话
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                self.equations[key] = value
                self.variables.append(key)
                self.values.append(value)
        return self.equations

    def get_value_by_sign(self, text):
        equations = self.__extract(text)
        return list(equations.values())

