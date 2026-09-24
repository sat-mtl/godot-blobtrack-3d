extends Node

class tasoeur extends RefCounted:
	var a = 4

func _ready() -> void:
	var aaaaarr := []
	for i in range(5):
		var enshorts = tasoeur.new()
		enshorts.a = i
		aaaaarr.append(enshorts)
	var b := aaaaarr.slice(0,3)
	for c in aaaaarr:
		c.a = 32
	for c in aaaaarr:
		print(c.a)
	for c in b:
		print(c.a)
