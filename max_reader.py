import struct

with open("out.txt", "rb") as test_file:
	with open("output.txt", 'w+') as output:
		data = test_file.read()
		words = str(data[:513],'ascii')
		output = open("guru99.txt","w+")
		output.write(words)
		print (len(data))
		while len(data)>4:
			ints = struct.unpack('=H', data[600:602])
			print(ints)
			# strInt = str(ints[1])
			# output.write(strInt)






