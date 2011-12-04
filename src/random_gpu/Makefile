#GNU Makefile to build Library librandom.a 
#
#04/22/02 C. Ludescher
#

ifneq ($(MAKELEVEL),0)
# if OBJ was defined for main Makefile one level down
ifeq ("${OBJ}",".")
	override OBJ=..
endif
endif

# define system dependent flags, etc.
-include ../share/Make.local
-include ../share/Make.flags

ifdef DEBUG
	F90FLAGS = $(DF90FLAGS)
	FFLAGS = $(DFFLAGS)
endif

ARC = $(OBJ)/lib/librandom.a
OBJDIR = $(OBJ)/obj/random


#=============================================================
# f90 modules
Mnams = random_f77.mod rngf77.mod  rngdef.mod rng.mod
Mobjs = rngmod.o rng.o random_f77.o random_gpu.o
MODS  = $(foreach file,$(Mobjs),$(ARC)($(file)))

ifeq ($(MODUP),Y)
 MODS0=$(foreach m,$(Mnams),$(shell  echo $(m) | tr 'a-z' 'A-Z'))
 MODULES=$(foreach m,$(MODS0),$(subst .MOD,.$(MODEXT),$(m)))
else
 MODULES = $(foreach m,$(Mnams),$(subst .mod,.$(MODEXT),$(m)))
endif 
 
# fortran archive members
F  = $(subst .f,.o, $(wildcard *.f)) $(subst .F,.o, $(wildcard *.F))
F90 = $(subst .f90,.o, $(wildcard *.f90)) $(subst .F90,.o, $(wildcard *.F90))
CU = $(subst .cu,.o, $(wildcard *.cu)) $(subst .cu,.o, $(wildcard *.cu))
FALL = $(F) $(F90) $(CU)

MEM = $(foreach file,$(FALL),$(ARC)($(file)))


.PHONY: clean realclean checkdir all

libs: FORCE $(ARC)
 
all: libs 
	@echo done  

$(OLDLIB): timestamp.inf
	@echo "--- DETECTED uflib.a source update"
	@echo "--- Re-Making $(ARC)"
	@$(MAKE) libs

FORCE:
	@test -d $(OBJ)/lib || mkdir -p $(OBJ)/lib
	@test -d $(OBJ)/mod || mkdir -p $(OBJ)/mod
	@test -d $(OBJ)/obj/random || mkdir -p $(OBJ)/obj/random
 
$(ARC): $(MODS) $(MEM)
	@$(RANLIB) $@ > /dev/null

$(ARC)(%.o): %.cu
	$(NVCC) -c $(NVCCFLAGS) -o $(OBJDIR)/$*.o $<
	$(make_ar)

#  compile fixed length fortran 90 code
$(ARC)(%.o): %.f
	$(FC) $(FFLAGS) $(MODFLAGS) -I./ $(INCFLAGS) $(OUT) $(OBJDIR)/$*.o $<
	$(make_ar)

# pre-process/compile fortran 77 code
$(ARC)(%.o): %.F
	$(F_proc)
	$(make_ar)

#  compile free form fortran 90 code
$(ARC)(%.o): %.f90
	$(FC90) $(FFLAGS) $(MODFLAGS) -I./ $(INCFLAGS) $(OUT) $(OBJDIR)/$*.o $<
	$(make_ar)

# pre-process/compile free form fortran 90 code
$(ARC)(%.o): %.F90
	$(F90_proc)
	$(make_ar)

#
#---------------------------------------------------
exec:
	@echo 

clean: 
	@rm -f $(OBJDIR)/*

realclean: clean
	@rm -f $(OBJ)/lib/*random.a
	@(cd $(MDIR); rm -f $(MODULES))

install:
	@test -d $(LIBDIR) || mkdir -p $(LIBDIR)
	@if test -f $(ARC); then \
	  umask 133; cp $(ARC) $(LIBDIR)/; \
	else test -f $(LIBDIR)/librandom.a || echo ">>> can not find $(ARC)";fi
	@test -d $(MODDIR) || mkdir -p $(MODDIR)
	@(cd $(MDIR); umask 133; cp $(MODULES) $(MODDIR)/)

uninstall:
	rm -f $(LIBDIR)/librandom.a
	@(cd $(MODDIR); rm -f $(MODULES))

