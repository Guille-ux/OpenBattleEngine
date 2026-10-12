class OrderType:
	GOTO_ORDER = 0
	ATTACK_ORDER = 1
	CANCEL_ORDER = 2
	REPORT_ORDER = 3

class Order:
	def __init__(self, type, arg):
		self.type = type # es un OrderType
		self.arg = arg