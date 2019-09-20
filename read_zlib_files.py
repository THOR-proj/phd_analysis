#!/usr/bin/python
import pylab as pl
import zlib
from struct import unpack

def read_wrf(infile,doaverage,xgridpoints,ygridpoints):
#	doaverage=0
#	xgridpoints=486
#	ygridpoints=498
	
	file_id=open(infile,'r')	
	file_binary=file_id.read(-1)
	
	def readbyte (byte_start):
	    return file_binary[byte_start:byte_start+4], byte_start+4
	
	def readcompressedbyte (byte_start):
	    return decompressed_level[byte_start:byte_start+4], byte_start+4
	
	def unpackint (inputstring):
	    return unpack('i',inputstring)
	
	def unpackflt (inputstring):
	    return unpack('f',inputstring)
	
	bite_now=0
	
	level_nbytes=[]
	for level in range(40):
	    temp, bite_now = readbyte(bite_now)
	    level_nbytes.append(unpackint(temp)[0])
	
	level_offset=[]
	for level in range(40):
	    temp, bite_now = readbyte(bite_now)
	    level_offset.append(unpackint(temp)[0])
	
	compressed_data=''
	for i in range(320,len(file_binary)):
	    temp, bite_now=readbyte(bite_now)
	    compressed_data=compressed_data+temp
	
	data=pl.zeros((40,xgridpoints*ygridpoints),"float")
	for level in range(40):
	    compressed_level=compressed_data[level_offset[level]:level_offset[level]+level_nbytes[level]]
	    decompressed_level=zlib.decompress(compressed_level)
	    bite_now=0
	    for i in range(int(xgridpoints*ygridpoints)):
		temp, bite_now =readcompressedbyte(bite_now)
		data[level,i]=unpackflt(temp)[0]


	return  data

