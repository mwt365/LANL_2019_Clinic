import struct

with open("GEN3CH_4_009.dig", "rb") as test_file:
	with open("output.txt", 'w+') as output:
		data = test_file.read()
		words = str(data[:513],'ascii')
		output = open("guru99.txt","w+")
		output.write(words)
		print (len(data))
		print(struct.calcsize('B'))
		# for i in range(int(len(data)/2)):
		# 	if (len(data) >2):
		intArray = []
		print(len(data))
		intArray += struct.unpack(">B", data[2:3])
		print(intArray)
		# strInt = str(ints[1])
		# output.write(strInt)






