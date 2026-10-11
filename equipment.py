class EquipmentSlot:
	def __init__(self, equipment, quantity):
		self.equipment = equipment
		self.quantity = quantity

class BaseEquipment:
	name = "Karabiner 98K"	
	max_range = 1.5
	breakthrough = 3
	soft_attack = 8
	hard_attack = 1
	supression = 3
	