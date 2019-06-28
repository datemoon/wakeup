
CXX=g++

GXX=g++


CXXFLAGS= -Ibin -g

all:libwakeup.a 


libwakeup.a:dnn_wakeup.o dtw.o
	ar -rcu $@ $^


%.o:%.cpp
	$(GXX) $(CXXFLAGS) -c -o $@ $<

.PHONY:

clean:
	rm -f *.o libwakeup.a
