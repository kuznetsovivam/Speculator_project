??8
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.0.0-beta12v2.0.0-beta0-16-g1d912138߶5
?
tech_dense_0_4/kernelVarHandleOp*
shape
:2*&
shared_nametech_dense_0_4/kernel*
dtype0*
_output_shapes
: 
?
)tech_dense_0_4/kernel/Read/ReadVariableOpReadVariableOptech_dense_0_4/kernel*(
_class
loc:@tech_dense_0_4/kernel*
dtype0*
_output_shapes

:2
~
tech_dense_0_4/biasVarHandleOp*
_output_shapes
: *
shape:2*$
shared_nametech_dense_0_4/bias*
dtype0
?
'tech_dense_0_4/bias/Read/ReadVariableOpReadVariableOptech_dense_0_4/bias*
_output_shapes
:2*&
_class
loc:@tech_dense_0_4/bias*
dtype0
?
tech_dense_3_4/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape
:2*&
shared_nametech_dense_3_4/kernel
?
)tech_dense_3_4/kernel/Read/ReadVariableOpReadVariableOptech_dense_3_4/kernel*(
_class
loc:@tech_dense_3_4/kernel*
dtype0*
_output_shapes

:2
~
tech_dense_3_4/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*$
shared_nametech_dense_3_4/bias
?
'tech_dense_3_4/bias/Read/ReadVariableOpReadVariableOptech_dense_3_4/bias*&
_class
loc:@tech_dense_3_4/bias*
dtype0*
_output_shapes
:
?
dense_pooling_5/kernelVarHandleOp*
_output_shapes
: *
shape
:2*'
shared_namedense_pooling_5/kernel*
dtype0
?
*dense_pooling_5/kernel/Read/ReadVariableOpReadVariableOpdense_pooling_5/kernel*)
_class
loc:@dense_pooling_5/kernel*
dtype0*
_output_shapes

:2
?
dense_pooling_5/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*%
shared_namedense_pooling_5/bias
?
(dense_pooling_5/bias/Read/ReadVariableOpReadVariableOpdense_pooling_5/bias*'
_class
loc:@dense_pooling_5/bias*
dtype0*
_output_shapes
:
?
dense_pooling_1_4/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape
:
*)
shared_namedense_pooling_1_4/kernel
?
,dense_pooling_1_4/kernel/Read/ReadVariableOpReadVariableOpdense_pooling_1_4/kernel*
_output_shapes

:
*+
_class!
loc:@dense_pooling_1_4/kernel*
dtype0
?
dense_pooling_1_4/biasVarHandleOp*'
shared_namedense_pooling_1_4/bias*
dtype0*
_output_shapes
: *
shape:

?
*dense_pooling_1_4/bias/Read/ReadVariableOpReadVariableOpdense_pooling_1_4/bias*)
_class
loc:@dense_pooling_1_4/bias*
dtype0*
_output_shapes
:

?
dense_out_4/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape
:
*#
shared_namedense_out_4/kernel
?
&dense_out_4/kernel/Read/ReadVariableOpReadVariableOpdense_out_4/kernel*%
_class
loc:@dense_out_4/kernel*
dtype0*
_output_shapes

:

x
dense_out_4/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*!
shared_namedense_out_4/bias
?
$dense_out_4/bias/Read/ReadVariableOpReadVariableOpdense_out_4/bias*#
_class
loc:@dense_out_4/bias*
dtype0*
_output_shapes
:
f
	Adam/iterVarHandleOp*
dtype0	*
_output_shapes
: *
shape: *
shared_name	Adam/iter
}
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_class
loc:@Adam/iter*
dtype0	*
_output_shapes
: 
j
Adam/beta_1VarHandleOp*
shape: *
shared_nameAdam/beta_1*
dtype0*
_output_shapes
: 
?
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: *
_class
loc:@Adam/beta_1
j
Adam/beta_2VarHandleOp*
shape: *
shared_nameAdam/beta_2*
dtype0*
_output_shapes
: 
?
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_name
Adam/decay
?
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 
x
Adam/learning_rateVarHandleOp*
shape: *#
shared_nameAdam/learning_rate*
dtype0*
_output_shapes
: 
?
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: *%
_class
loc:@Adam/learning_rate
{
lstm_0_4/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	?* 
shared_namelstm_0_4/kernel
?
#lstm_0_4/kernel/Read/ReadVariableOpReadVariableOplstm_0_4/kernel*
dtype0*
_output_shapes
:	?*"
_class
loc:@lstm_0_4/kernel
?
lstm_0_4/recurrent_kernelVarHandleOp**
shared_namelstm_0_4/recurrent_kernel*
dtype0*
_output_shapes
: *
shape:	2?
?
-lstm_0_4/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm_0_4/recurrent_kernel*,
_class"
 loc:@lstm_0_4/recurrent_kernel*
dtype0*
_output_shapes
:	2?
s
lstm_0_4/biasVarHandleOp*
shape:?*
shared_namelstm_0_4/bias*
dtype0*
_output_shapes
: 
?
!lstm_0_4/bias/Read/ReadVariableOpReadVariableOplstm_0_4/bias* 
_class
loc:@lstm_0_4/bias*
dtype0*
_output_shapes	
:?
z
lstm_1_4/kernelVarHandleOp*
shape
:2d* 
shared_namelstm_1_4/kernel*
dtype0*
_output_shapes
: 
?
#lstm_1_4/kernel/Read/ReadVariableOpReadVariableOplstm_1_4/kernel*"
_class
loc:@lstm_1_4/kernel*
dtype0*
_output_shapes

:2d
?
lstm_1_4/recurrent_kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape
:d**
shared_namelstm_1_4/recurrent_kernel
?
-lstm_1_4/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm_1_4/recurrent_kernel*,
_class"
 loc:@lstm_1_4/recurrent_kernel*
dtype0*
_output_shapes

:d
r
lstm_1_4/biasVarHandleOp*
shape:d*
shared_namelstm_1_4/bias*
dtype0*
_output_shapes
: 
?
!lstm_1_4/bias/Read/ReadVariableOpReadVariableOplstm_1_4/bias* 
_class
loc:@lstm_1_4/bias*
dtype0*
_output_shapes
:d
^
totalVarHandleOp*
shape: *
shared_nametotal*
dtype0*
_output_shapes
: 
q
total/Read/ReadVariableOpReadVariableOptotal*
_class

loc:@total*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shape: *
shared_namecount*
dtype0*
_output_shapes
: 
q
count/Read/ReadVariableOpReadVariableOpcount*
_class

loc:@count*
dtype0*
_output_shapes
: 
?
Adam/tech_dense_0_4/kernel/mVarHandleOp*
_output_shapes
: *
shape
:2*-
shared_nameAdam/tech_dense_0_4/kernel/m*
dtype0
?
0Adam/tech_dense_0_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/tech_dense_0_4/kernel/m*
_output_shapes

:2*/
_class%
#!loc:@Adam/tech_dense_0_4/kernel/m*
dtype0
?
Adam/tech_dense_0_4/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:2*+
shared_nameAdam/tech_dense_0_4/bias/m
?
.Adam/tech_dense_0_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/tech_dense_0_4/bias/m*
dtype0*
_output_shapes
:2*-
_class#
!loc:@Adam/tech_dense_0_4/bias/m
?
Adam/tech_dense_3_4/kernel/mVarHandleOp*-
shared_nameAdam/tech_dense_3_4/kernel/m*
dtype0*
_output_shapes
: *
shape
:2
?
0Adam/tech_dense_3_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/tech_dense_3_4/kernel/m*/
_class%
#!loc:@Adam/tech_dense_3_4/kernel/m*
dtype0*
_output_shapes

:2
?
Adam/tech_dense_3_4/bias/mVarHandleOp*
_output_shapes
: *
shape:*+
shared_nameAdam/tech_dense_3_4/bias/m*
dtype0
?
.Adam/tech_dense_3_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/tech_dense_3_4/bias/m*-
_class#
!loc:@Adam/tech_dense_3_4/bias/m*
dtype0*
_output_shapes
:
?
Adam/dense_pooling_5/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape
:2*.
shared_nameAdam/dense_pooling_5/kernel/m
?
1Adam/dense_pooling_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_pooling_5/kernel/m*0
_class&
$"loc:@Adam/dense_pooling_5/kernel/m*
dtype0*
_output_shapes

:2
?
Adam/dense_pooling_5/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:*,
shared_nameAdam/dense_pooling_5/bias/m
?
/Adam/dense_pooling_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_pooling_5/bias/m*
_output_shapes
:*.
_class$
" loc:@Adam/dense_pooling_5/bias/m*
dtype0
?
Adam/dense_pooling_1_4/kernel/mVarHandleOp*
shape
:
*0
shared_name!Adam/dense_pooling_1_4/kernel/m*
dtype0*
_output_shapes
: 
?
3Adam/dense_pooling_1_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_pooling_1_4/kernel/m*2
_class(
&$loc:@Adam/dense_pooling_1_4/kernel/m*
dtype0*
_output_shapes

:

?
Adam/dense_pooling_1_4/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:
*.
shared_nameAdam/dense_pooling_1_4/bias/m
?
1Adam/dense_pooling_1_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_pooling_1_4/bias/m*0
_class&
$"loc:@Adam/dense_pooling_1_4/bias/m*
dtype0*
_output_shapes
:

?
Adam/dense_out_4/kernel/mVarHandleOp**
shared_nameAdam/dense_out_4/kernel/m*
dtype0*
_output_shapes
: *
shape
:

?
-Adam/dense_out_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_out_4/kernel/m*,
_class"
 loc:@Adam/dense_out_4/kernel/m*
dtype0*
_output_shapes

:

?
Adam/dense_out_4/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:*(
shared_nameAdam/dense_out_4/bias/m
?
+Adam/dense_out_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_out_4/bias/m*
_output_shapes
:**
_class 
loc:@Adam/dense_out_4/bias/m*
dtype0
?
Adam/lstm_0_4/kernel/mVarHandleOp*
_output_shapes
: *
shape:	?*'
shared_nameAdam/lstm_0_4/kernel/m*
dtype0
?
*Adam/lstm_0_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm_0_4/kernel/m*
dtype0*
_output_shapes
:	?*)
_class
loc:@Adam/lstm_0_4/kernel/m
?
 Adam/lstm_0_4/recurrent_kernel/mVarHandleOp*1
shared_name" Adam/lstm_0_4/recurrent_kernel/m*
dtype0*
_output_shapes
: *
shape:	2?
?
4Adam/lstm_0_4/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_0_4/recurrent_kernel/m*3
_class)
'%loc:@Adam/lstm_0_4/recurrent_kernel/m*
dtype0*
_output_shapes
:	2?
?
Adam/lstm_0_4/bias/mVarHandleOp*
_output_shapes
: *
shape:?*%
shared_nameAdam/lstm_0_4/bias/m*
dtype0
?
(Adam/lstm_0_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_0_4/bias/m*'
_class
loc:@Adam/lstm_0_4/bias/m*
dtype0*
_output_shapes	
:?
?
Adam/lstm_1_4/kernel/mVarHandleOp*
shape
:2d*'
shared_nameAdam/lstm_1_4/kernel/m*
dtype0*
_output_shapes
: 
?
*Adam/lstm_1_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm_1_4/kernel/m*
_output_shapes

:2d*)
_class
loc:@Adam/lstm_1_4/kernel/m*
dtype0
?
 Adam/lstm_1_4/recurrent_kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape
:d*1
shared_name" Adam/lstm_1_4/recurrent_kernel/m
?
4Adam/lstm_1_4/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_1_4/recurrent_kernel/m*
dtype0*
_output_shapes

:d*3
_class)
'%loc:@Adam/lstm_1_4/recurrent_kernel/m
?
Adam/lstm_1_4/bias/mVarHandleOp*%
shared_nameAdam/lstm_1_4/bias/m*
dtype0*
_output_shapes
: *
shape:d
?
(Adam/lstm_1_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_1_4/bias/m*'
_class
loc:@Adam/lstm_1_4/bias/m*
dtype0*
_output_shapes
:d
?
Adam/tech_dense_0_4/kernel/vVarHandleOp*-
shared_nameAdam/tech_dense_0_4/kernel/v*
dtype0*
_output_shapes
: *
shape
:2
?
0Adam/tech_dense_0_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/tech_dense_0_4/kernel/v*
_output_shapes

:2*/
_class%
#!loc:@Adam/tech_dense_0_4/kernel/v*
dtype0
?
Adam/tech_dense_0_4/bias/vVarHandleOp*+
shared_nameAdam/tech_dense_0_4/bias/v*
dtype0*
_output_shapes
: *
shape:2
?
.Adam/tech_dense_0_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/tech_dense_0_4/bias/v*-
_class#
!loc:@Adam/tech_dense_0_4/bias/v*
dtype0*
_output_shapes
:2
?
Adam/tech_dense_3_4/kernel/vVarHandleOp*-
shared_nameAdam/tech_dense_3_4/kernel/v*
dtype0*
_output_shapes
: *
shape
:2
?
0Adam/tech_dense_3_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/tech_dense_3_4/kernel/v*
_output_shapes

:2*/
_class%
#!loc:@Adam/tech_dense_3_4/kernel/v*
dtype0
?
Adam/tech_dense_3_4/bias/vVarHandleOp*+
shared_nameAdam/tech_dense_3_4/bias/v*
dtype0*
_output_shapes
: *
shape:
?
.Adam/tech_dense_3_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/tech_dense_3_4/bias/v*
_output_shapes
:*-
_class#
!loc:@Adam/tech_dense_3_4/bias/v*
dtype0
?
Adam/dense_pooling_5/kernel/vVarHandleOp*
shape
:2*.
shared_nameAdam/dense_pooling_5/kernel/v*
dtype0*
_output_shapes
: 
?
1Adam/dense_pooling_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_pooling_5/kernel/v*
dtype0*
_output_shapes

:2*0
_class&
$"loc:@Adam/dense_pooling_5/kernel/v
?
Adam/dense_pooling_5/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:*,
shared_nameAdam/dense_pooling_5/bias/v
?
/Adam/dense_pooling_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_pooling_5/bias/v*.
_class$
" loc:@Adam/dense_pooling_5/bias/v*
dtype0*
_output_shapes
:
?
Adam/dense_pooling_1_4/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape
:
*0
shared_name!Adam/dense_pooling_1_4/kernel/v
?
3Adam/dense_pooling_1_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_pooling_1_4/kernel/v*2
_class(
&$loc:@Adam/dense_pooling_1_4/kernel/v*
dtype0*
_output_shapes

:

?
Adam/dense_pooling_1_4/bias/vVarHandleOp*
_output_shapes
: *
shape:
*.
shared_nameAdam/dense_pooling_1_4/bias/v*
dtype0
?
1Adam/dense_pooling_1_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_pooling_1_4/bias/v*0
_class&
$"loc:@Adam/dense_pooling_1_4/bias/v*
dtype0*
_output_shapes
:

?
Adam/dense_out_4/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape
:
**
shared_nameAdam/dense_out_4/kernel/v
?
-Adam/dense_out_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_out_4/kernel/v*,
_class"
 loc:@Adam/dense_out_4/kernel/v*
dtype0*
_output_shapes

:

?
Adam/dense_out_4/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:*(
shared_nameAdam/dense_out_4/bias/v
?
+Adam/dense_out_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_out_4/bias/v*
dtype0*
_output_shapes
:**
_class 
loc:@Adam/dense_out_4/bias/v
?
Adam/lstm_0_4/kernel/vVarHandleOp*
shape:	?*'
shared_nameAdam/lstm_0_4/kernel/v*
dtype0*
_output_shapes
: 
?
*Adam/lstm_0_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm_0_4/kernel/v*)
_class
loc:@Adam/lstm_0_4/kernel/v*
dtype0*
_output_shapes
:	?
?
 Adam/lstm_0_4/recurrent_kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:	2?*1
shared_name" Adam/lstm_0_4/recurrent_kernel/v
?
4Adam/lstm_0_4/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_0_4/recurrent_kernel/v*3
_class)
'%loc:@Adam/lstm_0_4/recurrent_kernel/v*
dtype0*
_output_shapes
:	2?
?
Adam/lstm_0_4/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:?*%
shared_nameAdam/lstm_0_4/bias/v
?
(Adam/lstm_0_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_0_4/bias/v*'
_class
loc:@Adam/lstm_0_4/bias/v*
dtype0*
_output_shapes	
:?
?
Adam/lstm_1_4/kernel/vVarHandleOp*'
shared_nameAdam/lstm_1_4/kernel/v*
dtype0*
_output_shapes
: *
shape
:2d
?
*Adam/lstm_1_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm_1_4/kernel/v*)
_class
loc:@Adam/lstm_1_4/kernel/v*
dtype0*
_output_shapes

:2d
?
 Adam/lstm_1_4/recurrent_kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape
:d*1
shared_name" Adam/lstm_1_4/recurrent_kernel/v
?
4Adam/lstm_1_4/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_1_4/recurrent_kernel/v*3
_class)
'%loc:@Adam/lstm_1_4/recurrent_kernel/v*
dtype0*
_output_shapes

:d
?
Adam/lstm_1_4/bias/vVarHandleOp*%
shared_nameAdam/lstm_1_4/bias/v*
dtype0*
_output_shapes
: *
shape:d
?
(Adam/lstm_1_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_1_4/bias/v*
dtype0*
_output_shapes
:d*'
_class
loc:@Adam/lstm_1_4/bias/v

NoOpNoOp
?^
ConstConst"/device:CPU:0*
_output_shapes
: *?^
value?^B?^ B?^
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
R
trainable_variables
regularization_losses
	variables
	keras_api
?

kernel
bias
_callable_losses
_eager_losses
trainable_variables
regularization_losses
	variables
	keras_api
{
_callable_losses
 _eager_losses
!trainable_variables
"regularization_losses
#	variables
$	keras_api
{
%_callable_losses
&_eager_losses
'trainable_variables
(regularization_losses
)	variables
*	keras_api
?
+cell
,
state_spec
-_callable_losses
._eager_losses
/trainable_variables
0regularization_losses
1	variables
2	keras_api
?

3kernel
4bias
5_callable_losses
6_eager_losses
7trainable_variables
8regularization_losses
9	variables
:	keras_api
?
;cell
<
state_spec
=_callable_losses
>_eager_losses
?trainable_variables
@regularization_losses
A	variables
B	keras_api
{
C_callable_losses
D_eager_losses
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
{
I_callable_losses
J_eager_losses
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
?

Okernel
Pbias
Q_callable_losses
R_eager_losses
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
?

Wkernel
Xbias
Y_callable_losses
Z_eager_losses
[trainable_variables
\regularization_losses
]	variables
^	keras_api
?

_kernel
`bias
a_callable_losses
b_eager_losses
ctrainable_variables
dregularization_losses
e	variables
f	keras_api
?
giter

hbeta_1

ibeta_2
	jdecay
klearning_ratem?m?3m?4m?Om?Pm?Wm?Xm?_m?`m?lm?mm?nm?om?pm?qm?v?v?3v?4v?Ov?Pv?Wv?Xv?_v?`v?lv?mv?nv?ov?pv?qv?
v
0
1
l2
m3
n4
35
46
o7
p8
q9
O10
P11
W12
X13
_14
`15
 
v
0
1
l2
m3
n4
35
46
o7
p8
q9
O10
P11
W12
X13
_14
`15
y
trainable_variables
rnon_trainable_variables
regularization_losses
	variables

slayers
tmetrics
 
 
 
 
y
unon_trainable_variables
trainable_variables
regularization_losses
	variables

vlayers
wmetrics
a_
VARIABLE_VALUEtech_dense_0_4/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEtech_dense_0_4/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
 

0
1
y
xnon_trainable_variables
trainable_variables
regularization_losses
	variables

ylayers
zmetrics
 
 
 
 
 
y
{non_trainable_variables
!trainable_variables
"regularization_losses
#	variables

|layers
}metrics
 
 
 
 
 
z
~non_trainable_variables
'trainable_variables
(regularization_losses
)	variables

layers
?metrics
?

lkernel
mrecurrent_kernel
nbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
 
 
 

l0
m1
n2
 

l0
m1
n2
|
?non_trainable_variables
/trainable_variables
0regularization_losses
1	variables
?layers
?metrics
a_
VARIABLE_VALUEtech_dense_3_4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEtech_dense_3_4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

30
41
 

30
41
|
?non_trainable_variables
7trainable_variables
8regularization_losses
9	variables
?layers
?metrics
?

okernel
precurrent_kernel
qbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
 
 
 

o0
p1
q2
 

o0
p1
q2
|
?non_trainable_variables
?trainable_variables
@regularization_losses
A	variables
?layers
?metrics
 
 
 
 
 
|
?non_trainable_variables
Etrainable_variables
Fregularization_losses
G	variables
?layers
?metrics
 
 
 
 
 
|
?non_trainable_variables
Ktrainable_variables
Lregularization_losses
M	variables
?layers
?metrics
b`
VARIABLE_VALUEdense_pooling_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEdense_pooling_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

O0
P1
 

O0
P1
|
?non_trainable_variables
Strainable_variables
Tregularization_losses
U	variables
?layers
?metrics
db
VARIABLE_VALUEdense_pooling_1_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEdense_pooling_1_4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

W0
X1
 

W0
X1
|
?non_trainable_variables
[trainable_variables
\regularization_losses
]	variables
?layers
?metrics
^\
VARIABLE_VALUEdense_out_4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_out_4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

_0
`1
 

_0
`1
|
?non_trainable_variables
ctrainable_variables
dregularization_losses
e	variables
?layers
?metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_0_4/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_0_4/recurrent_kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_0_4/bias0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_1_4/kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_1_4/recurrent_kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_1_4/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
 
V
0
1
2
3
4
5
6
7
	8

9
10
11

?0
 
 
 
 
 
 
 
 
 
 
 
 

l0
m1
n2
 

l0
m1
n2

?non_trainable_variables
?trainable_variables
?regularization_losses
?	variables
?layers
?metrics
 

+0
 
 
 
 

o0
p1
q2
 

o0
p1
q2

?non_trainable_variables
?trainable_variables
?regularization_losses
?	variables
?layers
?metrics
 

;0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?

?total

?count
?
_fn_kwargs
?_updates
?trainable_variables
?regularization_losses
?	variables
?	keras_api
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 

?0
?1

?non_trainable_variables
?trainable_variables
?regularization_losses
?	variables
?layers
?metrics

?0
?1
 
 
??
VARIABLE_VALUEAdam/tech_dense_0_4/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/tech_dense_0_4/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/tech_dense_3_4/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/tech_dense_3_4/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_pooling_5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/dense_pooling_5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_pooling_1_4/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_pooling_1_4/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/dense_out_4/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_out_4/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/lstm_0_4/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_0_4/recurrent_kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_0_4/bias/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/lstm_1_4/kernel/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_1_4/recurrent_kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_1_4/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/tech_dense_0_4/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/tech_dense_0_4/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/tech_dense_3_4/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/tech_dense_3_4/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_pooling_5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/dense_pooling_5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_pooling_1_4/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/dense_pooling_1_4/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/dense_out_4/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_out_4/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/lstm_0_4/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_0_4/recurrent_kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_0_4/bias/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/lstm_1_4/kernel/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_1_4/recurrent_kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/lstm_1_4/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0
?
serving_default_lstm_inputPlaceholder* 
shape:?????????0*
dtype0*+
_output_shapes
:?????????0
}
serving_default_tech_inputPlaceholder*
dtype0*'
_output_shapes
:?????????*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_inputserving_default_tech_inputtech_dense_0_4/kerneltech_dense_0_4/biastech_dense_3_4/kerneltech_dense_3_4/biaslstm_0_4/kernellstm_0_4/recurrent_kernellstm_0_4/biaslstm_1_4/kernellstm_1_4/recurrent_kernellstm_1_4/biasdense_pooling_5/kerneldense_pooling_5/biasdense_pooling_1_4/kerneldense_pooling_1_4/biasdense_out_4/kerneldense_out_4/bias*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????*,
f'R%
#__inference_signature_wrapper_77046
O
saver_filenamePlaceholder*
_output_shapes
: *
shape: *
dtype0
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)tech_dense_0_4/kernel/Read/ReadVariableOp'tech_dense_0_4/bias/Read/ReadVariableOp)tech_dense_3_4/kernel/Read/ReadVariableOp'tech_dense_3_4/bias/Read/ReadVariableOp*dense_pooling_5/kernel/Read/ReadVariableOp(dense_pooling_5/bias/Read/ReadVariableOp,dense_pooling_1_4/kernel/Read/ReadVariableOp*dense_pooling_1_4/bias/Read/ReadVariableOp&dense_out_4/kernel/Read/ReadVariableOp$dense_out_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp#lstm_0_4/kernel/Read/ReadVariableOp-lstm_0_4/recurrent_kernel/Read/ReadVariableOp!lstm_0_4/bias/Read/ReadVariableOp#lstm_1_4/kernel/Read/ReadVariableOp-lstm_1_4/recurrent_kernel/Read/ReadVariableOp!lstm_1_4/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp0Adam/tech_dense_0_4/kernel/m/Read/ReadVariableOp.Adam/tech_dense_0_4/bias/m/Read/ReadVariableOp0Adam/tech_dense_3_4/kernel/m/Read/ReadVariableOp.Adam/tech_dense_3_4/bias/m/Read/ReadVariableOp1Adam/dense_pooling_5/kernel/m/Read/ReadVariableOp/Adam/dense_pooling_5/bias/m/Read/ReadVariableOp3Adam/dense_pooling_1_4/kernel/m/Read/ReadVariableOp1Adam/dense_pooling_1_4/bias/m/Read/ReadVariableOp-Adam/dense_out_4/kernel/m/Read/ReadVariableOp+Adam/dense_out_4/bias/m/Read/ReadVariableOp*Adam/lstm_0_4/kernel/m/Read/ReadVariableOp4Adam/lstm_0_4/recurrent_kernel/m/Read/ReadVariableOp(Adam/lstm_0_4/bias/m/Read/ReadVariableOp*Adam/lstm_1_4/kernel/m/Read/ReadVariableOp4Adam/lstm_1_4/recurrent_kernel/m/Read/ReadVariableOp(Adam/lstm_1_4/bias/m/Read/ReadVariableOp0Adam/tech_dense_0_4/kernel/v/Read/ReadVariableOp.Adam/tech_dense_0_4/bias/v/Read/ReadVariableOp0Adam/tech_dense_3_4/kernel/v/Read/ReadVariableOp.Adam/tech_dense_3_4/bias/v/Read/ReadVariableOp1Adam/dense_pooling_5/kernel/v/Read/ReadVariableOp/Adam/dense_pooling_5/bias/v/Read/ReadVariableOp3Adam/dense_pooling_1_4/kernel/v/Read/ReadVariableOp1Adam/dense_pooling_1_4/bias/v/Read/ReadVariableOp-Adam/dense_out_4/kernel/v/Read/ReadVariableOp+Adam/dense_out_4/bias/v/Read/ReadVariableOp*Adam/lstm_0_4/kernel/v/Read/ReadVariableOp4Adam/lstm_0_4/recurrent_kernel/v/Read/ReadVariableOp(Adam/lstm_0_4/bias/v/Read/ReadVariableOp*Adam/lstm_1_4/kernel/v/Read/ReadVariableOp4Adam/lstm_1_4/recurrent_kernel/v/Read/ReadVariableOp(Adam/lstm_1_4/bias/v/Read/ReadVariableOpConst**
config_proto

GPU 

CPU2J 8*D
Tin=
;29	*
_output_shapes
: *,
_gradient_op_typePartitionedCall-77239*'
f"R 
__inference__traced_save_77238*
Tout
2
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametech_dense_0_4/kerneltech_dense_0_4/biastech_dense_3_4/kerneltech_dense_3_4/biasdense_pooling_5/kerneldense_pooling_5/biasdense_pooling_1_4/kerneldense_pooling_1_4/biasdense_out_4/kerneldense_out_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_0_4/kernellstm_0_4/recurrent_kernellstm_0_4/biaslstm_1_4/kernellstm_1_4/recurrent_kernellstm_1_4/biastotalcountAdam/tech_dense_0_4/kernel/mAdam/tech_dense_0_4/bias/mAdam/tech_dense_3_4/kernel/mAdam/tech_dense_3_4/bias/mAdam/dense_pooling_5/kernel/mAdam/dense_pooling_5/bias/mAdam/dense_pooling_1_4/kernel/mAdam/dense_pooling_1_4/bias/mAdam/dense_out_4/kernel/mAdam/dense_out_4/bias/mAdam/lstm_0_4/kernel/m Adam/lstm_0_4/recurrent_kernel/mAdam/lstm_0_4/bias/mAdam/lstm_1_4/kernel/m Adam/lstm_1_4/recurrent_kernel/mAdam/lstm_1_4/bias/mAdam/tech_dense_0_4/kernel/vAdam/tech_dense_0_4/bias/vAdam/tech_dense_3_4/kernel/vAdam/tech_dense_3_4/bias/vAdam/dense_pooling_5/kernel/vAdam/dense_pooling_5/bias/vAdam/dense_pooling_1_4/kernel/vAdam/dense_pooling_1_4/bias/vAdam/dense_out_4/kernel/vAdam/dense_out_4/bias/vAdam/lstm_0_4/kernel/v Adam/lstm_0_4/recurrent_kernel/vAdam/lstm_0_4/bias/vAdam/lstm_1_4/kernel/v Adam/lstm_1_4/recurrent_kernel/vAdam/lstm_1_4/bias/v*,
_gradient_op_typePartitionedCall-77417**
f%R#
!__inference__traced_restore_77416*
Tout
2**
config_proto

GPU 

CPU2J 8*
_output_shapes
: *C
Tin<
:28٨3
?G
?
__inference_standard_lstm_70653

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?whilec
transpose/permConst*
_output_shapes
:*!
valueB"          *
dtype0m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:0?????????2B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
valueB :
?????????*
dtype0*
_output_shapes
: ?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: ?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"????2   *
dtype0*
_output_shapes
:?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*'
_output_shapes
:?????????2*
Index0*
T0*
shrink_axis_mask?
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:2d{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0b
addAddMatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:?????????:?????????:?????????:?????????T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:?????????*
T0V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:?????????*
T0S
mulMulSigmoid_1:y:0init_c*'
_output_shapes
:?????????*
T0N
TanhTanhsplit:output:2*'
_output_shapes
:?????????*
T0U
mul_1MulSigmoid:y:0Tanh:y:0*'
_output_shapes
:?????????*
T0R
add_1Addmul:z:0	mul_1:z:0*'
_output_shapes
:?????????*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????K
Tanh_1Tanh	add_1:z:0*'
_output_shapes
:?????????*
T0Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????h
TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
valueB :
?????????*
dtype0?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
value	B :0T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *
T
2*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_70547*
bodyR
while_body_70548K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:?????????^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:?????????M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"????   *
dtype0*
_output_shapes
:?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:0?????????h
strided_slice_2/stackConst*
valueB:
?????????*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
valueB:*
dtype0?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*'
_output_shapes
:?????????*
Index0*
T0*
shrink_axis_maske
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????0[
runtimeConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ???
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:?????????*
T0?

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:?????????0?

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:?????????*
T0?

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:??????????

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*
	_noinline(*\
_input_shapesK
I:?????????02:?????????:?????????:::*=
api_implements+)lstm_ffcd560d-1075-4e30-88b0-2ea61de6018e*
api_preferred_deviceCPU2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?
b
F__inference_tech_relu_0_layer_call_and_return_conditional_losses_74774

inputs
identityF
ReluReluinputs*'
_output_shapes
:?????????2*
T0Z
IdentityIdentityRelu:activations:0*'
_output_shapes
:?????????2*
T0"
identityIdentity:output:0*&
_input_shapes
:?????????2:& "
 
_user_specified_nameinputs
?+
?
while_body_74337
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"????2   *
dtype0*
_output_shapes
:?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0time:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:?????????2?
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:2d?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0b
addAddMatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
T0T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:?????????*
T0V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????S
mulMulSigmoid_1:y:0init_c*'
_output_shapes
:?????????*
T0N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????R
add_1Addmul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:?????????*
T0K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*'
_output_shapes
:?????????*
T0?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemtensorarrayv2_1time	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: E
add_2Addtimeadd_2/y:output:0*
_output_shapes
: *
T0I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: S
add_3Addwhile_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: ?
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0?

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:??????????

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:  : : : : : : : : :	 :
 
?
G
+__inference_tech_relu_3_layer_call_fn_76757

inputs
identity?
PartitionedCallPartitionedCallinputs*'
_output_shapes
:?????????*
Tin
2*,
_gradient_op_typePartitionedCall-76754*O
fJRH
F__inference_tech_relu_3_layer_call_and_return_conditional_losses_76748*
Tout
2**
config_proto

GPU 

CPU2J 8`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?P
?
__forward_cudnn_lstm_76544

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*
T0l
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0;

zeros_likeConst*
valueBd*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
dtype0*
value	B : W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
?????????*
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0o
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
dtype0*
valueB:
?????????C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
T0*
Index0C
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
dtype0*
valueB:E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"-
transpose_9_permtranspose_9/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"!

identity_1Identity_1:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"-
transpose_8_permtranspose_8/perm:output:0"
identityIdentity:output:0"3
strided_slice_stackstrided_slice/stack:output:0"-
transpose_2_permtranspose_2/perm:output:0"'
concat_1_axisconcat_1/axis:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0"
concat_1concat_1_0:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0*
_input_shapes *=
api_implements+)lstm_d99063c4-b3b5-4528-b76f-dadf8da915f1*
api_preferred_deviceGPU*
	_noinline(*I
backward_function_name/-__inference___backward_cudnn_lstm_76542_767182
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?
?
(__inference_model_14_layer_call_fn_77014

lstm_input

tech_input"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
lstm_input
tech_inputstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17*,
_gradient_op_typePartitionedCall-76995*L
fGRE
C__inference_model_14_layer_call_and_return_conditional_losses_76994*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*}
_input_shapesl
j:?????????0:?????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : :* &
$
_user_specified_name
lstm_input:*&
$
_user_specified_name
tech_input: : : : : : : :	 :
 : 
?
?
while_cond_71976
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
G
LessLesstimeless_strided_slice*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : ::::  : : : : : : : : :	 :
 
?
?
&__inference_lstm_1_layer_call_fn_76738

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*,
_gradient_op_typePartitionedCall-76732*J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_76720*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????02:::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs
?5
?
C__inference_model_14_layer_call_and_return_conditional_losses_76994

inputs
inputs_1/
+tech_dense_0_statefulpartitionedcall_args_1/
+tech_dense_0_statefulpartitionedcall_args_2/
+tech_dense_3_statefulpartitionedcall_args_1/
+tech_dense_3_statefulpartitionedcall_args_2)
%lstm_0_statefulpartitionedcall_args_1)
%lstm_0_statefulpartitionedcall_args_2)
%lstm_0_statefulpartitionedcall_args_3)
%lstm_1_statefulpartitionedcall_args_1)
%lstm_1_statefulpartitionedcall_args_2)
%lstm_1_statefulpartitionedcall_args_30
,dense_pooling_statefulpartitionedcall_args_10
,dense_pooling_statefulpartitionedcall_args_22
.dense_pooling_1_statefulpartitionedcall_args_12
.dense_pooling_1_statefulpartitionedcall_args_2,
(dense_out_statefulpartitionedcall_args_1,
(dense_out_statefulpartitionedcall_args_2
identity??!dense_out/StatefulPartitionedCall?%dense_pooling/StatefulPartitionedCall?'dense_pooling_1/StatefulPartitionedCall?lstm_0/StatefulPartitionedCall?lstm_1/StatefulPartitionedCall?$tech_dense_0/StatefulPartitionedCall?$tech_dense_3/StatefulPartitionedCall?
$tech_dense_0/StatefulPartitionedCallStatefulPartitionedCallinputs_1+tech_dense_0_statefulpartitionedcall_args_1+tech_dense_0_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-74758*P
fKRI
G__inference_tech_dense_0_layer_call_and_return_conditional_losses_74752*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????2?
tech_relu_0/PartitionedCallPartitionedCall-tech_dense_0/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-74780*O
fJRH
F__inference_tech_relu_0_layer_call_and_return_conditional_losses_74774*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????2?
lstm_input/PartitionedCallPartitionedCallinputs*N
fIRG
E__inference_lstm_input_layer_call_and_return_conditional_losses_74791*
Tout
2**
config_proto

GPU 

CPU2J 8*+
_output_shapes
:?????????0*
Tin
2*,
_gradient_op_typePartitionedCall-74797?
$tech_dense_3/StatefulPartitionedCallStatefulPartitionedCall$tech_relu_0/PartitionedCall:output:0+tech_dense_3_statefulpartitionedcall_args_1+tech_dense_3_statefulpartitionedcall_args_2*P
fKRI
G__inference_tech_dense_3_layer_call_and_return_conditional_losses_74814*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-74820?
lstm_0/StatefulPartitionedCallStatefulPartitionedCall#lstm_input/PartitionedCall:output:0%lstm_0_statefulpartitionedcall_args_1%lstm_0_statefulpartitionedcall_args_2%lstm_0_statefulpartitionedcall_args_3*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:?????????02*,
_gradient_op_typePartitionedCall-75776*J
fERC
A__inference_lstm_0_layer_call_and_return_conditional_losses_75764?
lstm_1/StatefulPartitionedCallStatefulPartitionedCall'lstm_0/StatefulPartitionedCall:output:0%lstm_1_statefulpartitionedcall_args_1%lstm_1_statefulpartitionedcall_args_2%lstm_1_statefulpartitionedcall_args_3**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-76732*J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_76720*
Tout
2?
tech_relu_3/PartitionedCallPartitionedCall-tech_dense_3/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-76754*O
fJRH
F__inference_tech_relu_3_layer_call_and_return_conditional_losses_76748*
Tout
2?
concatenate/PartitionedCallPartitionedCall'lstm_1/StatefulPartitionedCall:output:0$tech_relu_3/PartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????2*
Tin
2*,
_gradient_op_typePartitionedCall-76775*O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_76768*
Tout
2?
%dense_pooling/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0,dense_pooling_statefulpartitionedcall_args_1,dense_pooling_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-76799*Q
fLRJ
H__inference_dense_pooling_layer_call_and_return_conditional_losses_76793*
Tout
2?
'dense_pooling_1/StatefulPartitionedCallStatefulPartitionedCall.dense_pooling/StatefulPartitionedCall:output:0.dense_pooling_1_statefulpartitionedcall_args_1.dense_pooling_1_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????
*,
_gradient_op_typePartitionedCall-76827*S
fNRL
J__inference_dense_pooling_1_layer_call_and_return_conditional_losses_76821*
Tout
2?
!dense_out/StatefulPartitionedCallStatefulPartitionedCall0dense_pooling_1/StatefulPartitionedCall:output:0(dense_out_statefulpartitionedcall_args_1(dense_out_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-76855*M
fHRF
D__inference_dense_out_layer_call_and_return_conditional_losses_76849*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:??????????
IdentityIdentity*dense_out/StatefulPartitionedCall:output:0"^dense_out/StatefulPartitionedCall&^dense_pooling/StatefulPartitionedCall(^dense_pooling_1/StatefulPartitionedCall^lstm_0/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall%^tech_dense_0/StatefulPartitionedCall%^tech_dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*}
_input_shapesl
j:?????????0:?????????::::::::::::::::2N
%dense_pooling/StatefulPartitionedCall%dense_pooling/StatefulPartitionedCall2F
!dense_out/StatefulPartitionedCall!dense_out/StatefulPartitionedCall2@
lstm_0/StatefulPartitionedCalllstm_0/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2L
$tech_dense_0/StatefulPartitionedCall$tech_dense_0/StatefulPartitionedCall2L
$tech_dense_3/StatefulPartitionedCall$tech_dense_3/StatefulPartitionedCall2R
'dense_pooling_1/StatefulPartitionedCall'dense_pooling_1/StatefulPartitionedCall: :& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs: : : : : : : :	 :
 : : : : : : 
?>
?
__inference_cudnn_lstm_72667

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*
dtype0*!
valueB"          @
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
dtype0*
value	B :i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueB?*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
	num_split*
T0>
Const_3Const*
valueB:
?????????*
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
dtype0*
valueB"       L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0k
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
Index0*
T0I
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
dtype0*
valueB:?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
Index0*
T0C
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
	_noinline(*
_input_shapes *=
api_implements+)lstm_a73b20f0-c393-4dea-a4aa-de0c2ac2517c*
api_preferred_deviceGPU26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?	
?
J__inference_dense_pooling_1_layer_call_and_return_conditional_losses_76821

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?+
?
while_body_75858
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"????2   *
dtype0*
_output_shapes
:?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0time:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:?????????2?
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:2d?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????db
addAddMatMul:product:0MatMul_1:product:0*'
_output_shapes
:?????????d*
T0?
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:?????????*
T0S
mulMulSigmoid_1:y:0init_c*'
_output_shapes
:?????????*
T0N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????R
add_1Addmul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????K
Tanh_1Tanh	add_1:z:0*'
_output_shapes
:?????????*
T0Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:??????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemtensorarrayv2_1time	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: E
add_2Addtimeadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: S
add_3Addwhile_loop_counteradd_3/y:output:0*
_output_shapes
: *
T0?
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:?????????*
T0?

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:  : : : : : : : : :	 :
 
?
?
&__inference_lstm_1_layer_call_fn_74737

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*,
_gradient_op_typePartitionedCall-74731*J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_74730*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
?+
?
while_body_72453
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"????   *
dtype0*
_output_shapes
:?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0time:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:??????????
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	??
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	2?n
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
addAddMatMul:product:0MatMul_1:product:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:?????????2*
T0V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:?????????2*
T0S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2N
TanhTanhsplit:output:2*'
_output_shapes
:?????????2*
T0U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2R
add_1Addmul:z:0	mul_1:z:0*'
_output_shapes
:?????????2*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????2Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*'
_output_shapes
:?????????2*
T0?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemtensorarrayv2_1time	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
_output_shapes
: *
value	B :*
dtype0E
add_2Addtimeadd_2/y:output:0*
_output_shapes
: *
T0I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: S
add_3Addwhile_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: ?
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0?

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:?????????2*
T0?

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:
 :  : : : : : : : : :	 
?H
?
__inference_standard_lstm_73964

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?whilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*4
_output_shapes"
 :??????????????????2*
T0B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
valueB :
?????????*
dtype0*
_output_shapes
: ?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: ?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"????2   *
dtype0*
_output_shapes
:?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:?????????2*
T0*
Index0?
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:2d{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0b
addAddMatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
T0T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:?????????*
T0V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:?????????*
T0S
mulMulSigmoid_1:y:0init_c*'
_output_shapes
:?????????*
T0N
TanhTanhsplit:output:2*'
_output_shapes
:?????????*
T0U
mul_1MulSigmoid:y:0Tanh:y:0*'
_output_shapes
:?????????*
T0R
add_1Addmul:z:0	mul_1:z:0*'
_output_shapes
:?????????*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*'
_output_shapes
:?????????*
T0h
TensorArrayV2_1/element_shapeConst*
valueB :
?????????*
dtype0*
_output_shapes
: ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
?????????*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
bodyR
while_body_73861*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *
T
2*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_73860K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:?????????*
T0^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:?????????M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
valueB"????   *
dtype0?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :??????????????????h
strided_slice_2/stackConst*
valueB:
?????????*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:?????????*
Index0*
T0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*4
_output_shapes"
 :??????????????????*
T0[
runtimeConst"/device:CPU:0*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:?????????*
T0?

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*4
_output_shapes"
 :??????????????????*
T0?

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:?????????*
T0?

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:??????????

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
_output_shapes
: *
T0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*e
_input_shapesT
R:??????????????????2:?????????:?????????:::*=
api_implements+)lstm_f5fde02f-8c16-4750-9be2-71a2dc99d757*
api_preferred_deviceCPU*
	_noinline(20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?H
?
__inference_standard_lstm_73029

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?whilec
transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2B
ShapeShapetranspose:y:0*
_output_shapes
:*
T0]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: ?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"????2   *
dtype0*
_output_shapes
:?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:?????????2?
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:2d{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????db
addAddMatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
T0T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:?????????*
T0V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:?????????*
T0S
mulMulSigmoid_1:y:0init_c*'
_output_shapes
:?????????*
T0N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????U
mul_1MulSigmoid:y:0Tanh:y:0*'
_output_shapes
:?????????*
T0R
add_1Addmul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????K
Tanh_1Tanh	add_1:z:0*'
_output_shapes
:?????????*
T0Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????h
TensorArrayV2_1/element_shapeConst*
valueB :
?????????*
dtype0*
_output_shapes
: ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
?????????*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
bodyR
while_body_72926*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *
T
2*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_72925K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:?????????^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:?????????M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"????   *
dtype0*
_output_shapes
:?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :??????????????????h
strided_slice_2/stackConst*
valueB:
?????????*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:?????????e
transpose_1/permConst*
_output_shapes
:*!
valueB"          *
dtype0?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????[
runtimeConst"/device:CPU:0*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:?????????*
T0?

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*4
_output_shapes"
 :??????????????????*
T0?

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:??????????

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:?????????*
T0?

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*e
_input_shapesT
R:??????????????????2:?????????:?????????:::*=
api_implements+)lstm_3f27c04b-652c-4bbb-81f8-497acfa4408c*
api_preferred_deviceCPU*
	_noinline(2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?P
?
__forward_cudnn_lstm_76077

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
dtype0*
value	B :;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0;

zeros_likeConst*
valueBd*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
	num_split*
T0>
Const_3Const*
valueB:
?????????*
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0o
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
dtype0*
valueB:?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"
concat_1concat_1_0:output:0"-
transpose_9_permtranspose_9/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"-
transpose_3_permtranspose_3/perm:output:0"!

identity_1Identity_1:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"-
transpose_8_permtranspose_8/perm:output:0"
identityIdentity:output:0"-
transpose_2_permtranspose_2/perm:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"'
concat_1_axisconcat_1/axis:output:0"
	transposetranspose_0:y:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_7_permtranspose_7/perm:output:0"#
concat_axisconcat/axis:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0")
transpose_permtranspose/perm:output:0"!

expanddimsExpandDims:output:0"-
transpose_4_permtranspose_4/perm:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"%
expanddims_1ExpandDims_1:output:0*
	_noinline(*I
backward_function_name/-__inference___backward_cudnn_lstm_76075_76251*
_input_shapes *=
api_implements+)lstm_07d2f4a3-a6ab-4ba4-8910-49922c38a5c9*
api_preferred_deviceGPU26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c
?P
?
__forward_cudnn_lstm_74078

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
dtype0*
value	B : D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*
T01
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*
T0l
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0;

zeros_likeConst*
valueBd*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
dtype0*
valueB:
?????????E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
dtype0*
valueB"       N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
dtype0*
valueB"       N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0o
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
dtype0*
valueB:?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
Index0*
T0C
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*
Index0*
T0C
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"-
transpose_8_permtranspose_8/perm:output:0"
identityIdentity:output:0"-
transpose_2_permtranspose_2/perm:output:0"3
strided_slice_stackstrided_slice/stack:output:0"'
concat_1_axisconcat_1/axis:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"%
expanddims_1ExpandDims_1:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"
concat_1concat_1_0:output:0"-
transpose_9_permtranspose_9/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"-
transpose_3_permtranspose_3/perm:output:0*
	_noinline(*I
backward_function_name/-__inference___backward_cudnn_lstm_74076_74252*
_input_shapes *=
api_implements+)lstm_f5fde02f-8c16-4750-9be2-71a2dc99d757*
api_preferred_deviceGPU20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel
?+
?
while_body_74902
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"????   *
dtype0*
_output_shapes
:?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0time:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:??????????
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	??
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	2?n
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
addAddMatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:???????????
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2N
TanhTanhsplit:output:2*'
_output_shapes
:?????????2*
T0U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2R
add_1Addmul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????2Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemtensorarrayv2_1time	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: E
add_2Addtimeadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: S
add_3Addwhile_loop_counteradd_3/y:output:0*
_output_shapes
: *
T0?
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0?

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0?

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:?????????2*
T0?

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:  : : : : : : : : :	 :
 
?
U
+__inference_concatenate_layer_call_fn_76778

inputs
inputs_1
identity?
PartitionedCallPartitionedCallinputsinputs_1*O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_76768*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????2*
Tin
2*,
_gradient_op_typePartitionedCall-76775`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:?????????2*
T0"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?f
?
__inference__traced_save_77238
file_prefix4
0savev2_tech_dense_0_4_kernel_read_readvariableop2
.savev2_tech_dense_0_4_bias_read_readvariableop4
0savev2_tech_dense_3_4_kernel_read_readvariableop2
.savev2_tech_dense_3_4_bias_read_readvariableop5
1savev2_dense_pooling_5_kernel_read_readvariableop3
/savev2_dense_pooling_5_bias_read_readvariableop7
3savev2_dense_pooling_1_4_kernel_read_readvariableop5
1savev2_dense_pooling_1_4_bias_read_readvariableop1
-savev2_dense_out_4_kernel_read_readvariableop/
+savev2_dense_out_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop.
*savev2_lstm_0_4_kernel_read_readvariableop8
4savev2_lstm_0_4_recurrent_kernel_read_readvariableop,
(savev2_lstm_0_4_bias_read_readvariableop.
*savev2_lstm_1_4_kernel_read_readvariableop8
4savev2_lstm_1_4_recurrent_kernel_read_readvariableop,
(savev2_lstm_1_4_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop;
7savev2_adam_tech_dense_0_4_kernel_m_read_readvariableop9
5savev2_adam_tech_dense_0_4_bias_m_read_readvariableop;
7savev2_adam_tech_dense_3_4_kernel_m_read_readvariableop9
5savev2_adam_tech_dense_3_4_bias_m_read_readvariableop<
8savev2_adam_dense_pooling_5_kernel_m_read_readvariableop:
6savev2_adam_dense_pooling_5_bias_m_read_readvariableop>
:savev2_adam_dense_pooling_1_4_kernel_m_read_readvariableop<
8savev2_adam_dense_pooling_1_4_bias_m_read_readvariableop8
4savev2_adam_dense_out_4_kernel_m_read_readvariableop6
2savev2_adam_dense_out_4_bias_m_read_readvariableop5
1savev2_adam_lstm_0_4_kernel_m_read_readvariableop?
;savev2_adam_lstm_0_4_recurrent_kernel_m_read_readvariableop3
/savev2_adam_lstm_0_4_bias_m_read_readvariableop5
1savev2_adam_lstm_1_4_kernel_m_read_readvariableop?
;savev2_adam_lstm_1_4_recurrent_kernel_m_read_readvariableop3
/savev2_adam_lstm_1_4_bias_m_read_readvariableop;
7savev2_adam_tech_dense_0_4_kernel_v_read_readvariableop9
5savev2_adam_tech_dense_0_4_bias_v_read_readvariableop;
7savev2_adam_tech_dense_3_4_kernel_v_read_readvariableop9
5savev2_adam_tech_dense_3_4_bias_v_read_readvariableop<
8savev2_adam_dense_pooling_5_kernel_v_read_readvariableop:
6savev2_adam_dense_pooling_5_bias_v_read_readvariableop>
:savev2_adam_dense_pooling_1_4_kernel_v_read_readvariableop<
8savev2_adam_dense_pooling_1_4_bias_v_read_readvariableop8
4savev2_adam_dense_out_4_kernel_v_read_readvariableop6
2savev2_adam_dense_out_4_bias_v_read_readvariableop5
1savev2_adam_lstm_0_4_kernel_v_read_readvariableop?
;savev2_adam_lstm_0_4_recurrent_kernel_v_read_readvariableop3
/savev2_adam_lstm_0_4_bias_v_read_readvariableop5
1savev2_adam_lstm_1_4_kernel_v_read_readvariableop?
;savev2_adam_lstm_1_4_recurrent_kernel_v_read_readvariableop3
/savev2_adam_lstm_1_4_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_e38751bee75c4552be75bcf0e8cfb5c1/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*?
value?B?7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:7?
SaveV2/shape_and_slicesConst"/device:CPU:0*?
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:7?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_tech_dense_0_4_kernel_read_readvariableop.savev2_tech_dense_0_4_bias_read_readvariableop0savev2_tech_dense_3_4_kernel_read_readvariableop.savev2_tech_dense_3_4_bias_read_readvariableop1savev2_dense_pooling_5_kernel_read_readvariableop/savev2_dense_pooling_5_bias_read_readvariableop3savev2_dense_pooling_1_4_kernel_read_readvariableop1savev2_dense_pooling_1_4_bias_read_readvariableop-savev2_dense_out_4_kernel_read_readvariableop+savev2_dense_out_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop*savev2_lstm_0_4_kernel_read_readvariableop4savev2_lstm_0_4_recurrent_kernel_read_readvariableop(savev2_lstm_0_4_bias_read_readvariableop*savev2_lstm_1_4_kernel_read_readvariableop4savev2_lstm_1_4_recurrent_kernel_read_readvariableop(savev2_lstm_1_4_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop7savev2_adam_tech_dense_0_4_kernel_m_read_readvariableop5savev2_adam_tech_dense_0_4_bias_m_read_readvariableop7savev2_adam_tech_dense_3_4_kernel_m_read_readvariableop5savev2_adam_tech_dense_3_4_bias_m_read_readvariableop8savev2_adam_dense_pooling_5_kernel_m_read_readvariableop6savev2_adam_dense_pooling_5_bias_m_read_readvariableop:savev2_adam_dense_pooling_1_4_kernel_m_read_readvariableop8savev2_adam_dense_pooling_1_4_bias_m_read_readvariableop4savev2_adam_dense_out_4_kernel_m_read_readvariableop2savev2_adam_dense_out_4_bias_m_read_readvariableop1savev2_adam_lstm_0_4_kernel_m_read_readvariableop;savev2_adam_lstm_0_4_recurrent_kernel_m_read_readvariableop/savev2_adam_lstm_0_4_bias_m_read_readvariableop1savev2_adam_lstm_1_4_kernel_m_read_readvariableop;savev2_adam_lstm_1_4_recurrent_kernel_m_read_readvariableop/savev2_adam_lstm_1_4_bias_m_read_readvariableop7savev2_adam_tech_dense_0_4_kernel_v_read_readvariableop5savev2_adam_tech_dense_0_4_bias_v_read_readvariableop7savev2_adam_tech_dense_3_4_kernel_v_read_readvariableop5savev2_adam_tech_dense_3_4_bias_v_read_readvariableop8savev2_adam_dense_pooling_5_kernel_v_read_readvariableop6savev2_adam_dense_pooling_5_bias_v_read_readvariableop:savev2_adam_dense_pooling_1_4_kernel_v_read_readvariableop8savev2_adam_dense_pooling_1_4_bias_v_read_readvariableop4savev2_adam_dense_out_4_kernel_v_read_readvariableop2savev2_adam_dense_out_4_bias_v_read_readvariableop1savev2_adam_lstm_0_4_kernel_v_read_readvariableop;savev2_adam_lstm_0_4_recurrent_kernel_v_read_readvariableop/savev2_adam_lstm_0_4_bias_v_read_readvariableop1savev2_adam_lstm_1_4_kernel_v_read_readvariableop;savev2_adam_lstm_1_4_recurrent_kernel_v_read_readvariableop/savev2_adam_lstm_1_4_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *E
dtypes;
927	h
ShardedFilename_1/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B :?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 ?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :2:2:2::2::
:
:
:: : : : : :	?:	2?:?:2d:d:d: : :2:2:2::2::
:
:
::	?:	2?:?:2d:d:d:2:2:2::2::
:
:
::	?:	2?:?:2d:d:d: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 :7 :8 
?5
?
C__inference_model_14_layer_call_and_return_conditional_losses_76901

lstm_input

tech_input/
+tech_dense_0_statefulpartitionedcall_args_1/
+tech_dense_0_statefulpartitionedcall_args_2/
+tech_dense_3_statefulpartitionedcall_args_1/
+tech_dense_3_statefulpartitionedcall_args_2)
%lstm_0_statefulpartitionedcall_args_1)
%lstm_0_statefulpartitionedcall_args_2)
%lstm_0_statefulpartitionedcall_args_3)
%lstm_1_statefulpartitionedcall_args_1)
%lstm_1_statefulpartitionedcall_args_2)
%lstm_1_statefulpartitionedcall_args_30
,dense_pooling_statefulpartitionedcall_args_10
,dense_pooling_statefulpartitionedcall_args_22
.dense_pooling_1_statefulpartitionedcall_args_12
.dense_pooling_1_statefulpartitionedcall_args_2,
(dense_out_statefulpartitionedcall_args_1,
(dense_out_statefulpartitionedcall_args_2
identity??!dense_out/StatefulPartitionedCall?%dense_pooling/StatefulPartitionedCall?'dense_pooling_1/StatefulPartitionedCall?lstm_0/StatefulPartitionedCall?lstm_1/StatefulPartitionedCall?$tech_dense_0/StatefulPartitionedCall?$tech_dense_3/StatefulPartitionedCall?
$tech_dense_0/StatefulPartitionedCallStatefulPartitionedCall
tech_input+tech_dense_0_statefulpartitionedcall_args_1+tech_dense_0_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????2*,
_gradient_op_typePartitionedCall-74758*P
fKRI
G__inference_tech_dense_0_layer_call_and_return_conditional_losses_74752*
Tout
2?
tech_relu_0/PartitionedCallPartitionedCall-tech_dense_0/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-74780*O
fJRH
F__inference_tech_relu_0_layer_call_and_return_conditional_losses_74774*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????2?
lstm_input_1/PartitionedCallPartitionedCall
lstm_input*
Tin
2*+
_output_shapes
:?????????0*,
_gradient_op_typePartitionedCall-74797*N
fIRG
E__inference_lstm_input_layer_call_and_return_conditional_losses_74791*
Tout
2**
config_proto

GPU 

CPU2J 8?
$tech_dense_3/StatefulPartitionedCallStatefulPartitionedCall$tech_relu_0/PartitionedCall:output:0+tech_dense_3_statefulpartitionedcall_args_1+tech_dense_3_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-74820*P
fKRI
G__inference_tech_dense_3_layer_call_and_return_conditional_losses_74814*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:??????????
lstm_0/StatefulPartitionedCallStatefulPartitionedCall%lstm_input_1/PartitionedCall:output:0%lstm_0_statefulpartitionedcall_args_1%lstm_0_statefulpartitionedcall_args_2%lstm_0_statefulpartitionedcall_args_3**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:?????????02*,
_gradient_op_typePartitionedCall-75776*J
fERC
A__inference_lstm_0_layer_call_and_return_conditional_losses_75764*
Tout
2?
lstm_1/StatefulPartitionedCallStatefulPartitionedCall'lstm_0/StatefulPartitionedCall:output:0%lstm_1_statefulpartitionedcall_args_1%lstm_1_statefulpartitionedcall_args_2%lstm_1_statefulpartitionedcall_args_3**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-76732*J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_76720*
Tout
2?
tech_relu_3/PartitionedCallPartitionedCall-tech_dense_3/StatefulPartitionedCall:output:0*
Tin
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-76754*O
fJRH
F__inference_tech_relu_3_layer_call_and_return_conditional_losses_76748*
Tout
2**
config_proto

GPU 

CPU2J 8?
concatenate/PartitionedCallPartitionedCall'lstm_1/StatefulPartitionedCall:output:0$tech_relu_3/PartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????2*,
_gradient_op_typePartitionedCall-76775*O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_76768*
Tout
2?
%dense_pooling/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0,dense_pooling_statefulpartitionedcall_args_1,dense_pooling_statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-76799*Q
fLRJ
H__inference_dense_pooling_layer_call_and_return_conditional_losses_76793?
'dense_pooling_1/StatefulPartitionedCallStatefulPartitionedCall.dense_pooling/StatefulPartitionedCall:output:0.dense_pooling_1_statefulpartitionedcall_args_1.dense_pooling_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-76827*S
fNRL
J__inference_dense_pooling_1_layer_call_and_return_conditional_losses_76821*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????
*
Tin
2?
!dense_out/StatefulPartitionedCallStatefulPartitionedCall0dense_pooling_1/StatefulPartitionedCall:output:0(dense_out_statefulpartitionedcall_args_1(dense_out_statefulpartitionedcall_args_2*'
_output_shapes
:?????????*
Tin
2*,
_gradient_op_typePartitionedCall-76855*M
fHRF
D__inference_dense_out_layer_call_and_return_conditional_losses_76849*
Tout
2**
config_proto

GPU 

CPU2J 8?
IdentityIdentity*dense_out/StatefulPartitionedCall:output:0"^dense_out/StatefulPartitionedCall&^dense_pooling/StatefulPartitionedCall(^dense_pooling_1/StatefulPartitionedCall^lstm_0/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall%^tech_dense_0/StatefulPartitionedCall%^tech_dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*}
_input_shapesl
j:?????????0:?????????::::::::::::::::2R
'dense_pooling_1/StatefulPartitionedCall'dense_pooling_1/StatefulPartitionedCall2N
%dense_pooling/StatefulPartitionedCall%dense_pooling/StatefulPartitionedCall2F
!dense_out/StatefulPartitionedCall!dense_out/StatefulPartitionedCall2@
lstm_0/StatefulPartitionedCalllstm_0/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2L
$tech_dense_0/StatefulPartitionedCall$tech_dense_0/StatefulPartitionedCall2L
$tech_dense_3/StatefulPartitionedCall$tech_dense_3/StatefulPartitionedCall:
 : : : : : : : :* &
$
_user_specified_name
lstm_input:*&
$
_user_specified_name
tech_input: : : : : : : :	 
??
?
-__inference___backward_cudnn_lstm_70765_70941
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis&
"gradients_transpose_grad_transpose%
!gradients_expanddims_grad_reshape'
#gradients_expanddims_1_grad_reshape
gradients_split_grad_concat!
gradients_split_1_grad_concat!
gradients_concat_grad_slice_1?^
gradients/grad_ys_0Identityplaceholder*'
_output_shapes
:?????????*
T0d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:?????????0`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????`
gradients/grad_ys_3Identityplaceholder_3*'
_output_shapes
:?????????*
T0O
gradients/grad_ys_4Identityplaceholder_4*
_output_shapes
: *
T0}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
shrink_axis_mask*+
_output_shapes
:0?????????*
Index0*
T0?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*+
_output_shapes
:0?????????*
T0?
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
_output_shapes
:*
T0?
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
shrink_axis_mask*+
_output_shapes
:?????????*
T0*
Index0?
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*
T0*
Index0*
shrink_axis_mask*+
_output_shapes
:??????????
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*+
_output_shapes
:0?????????*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGradc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*`
_output_shapesN
L:0?????????2:?????????:?????????:?<*
T0?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:?????????02u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:?????????^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: ?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:?	j
gradients/concat_1_grad/Shape_1Const*
valueB:?	*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_2Const*
valueB:?	*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_3Const*
valueB:?	*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_4Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_5Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
valueB:?*
dtype0j
gradients/concat_1_grad/Shape_7Const*
valueB:?*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_8Const*
valueB:*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_11Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_12Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_13Const*
dtype0*
_output_shapes
:*
valueB:j
 gradients/concat_1_grad/Shape_14Const*
dtype0*
_output_shapes
:*
valueB:j
 gradients/concat_1_grad/Shape_15Const*
valueB:*
dtype0*
_output_shapes
:?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
_output_shapes	
:?	*
Index0*
T0?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:?	?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
_output_shapes	
:?	*
Index0*
T0?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
_output_shapes	
:?	*
Index0*
T0?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
_output_shapes
:*
Index0*
T0?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
_output_shapes
:*
Index0*
T0?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
_output_shapes
:*
Index0*
T0?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
_output_shapes
:*
Index0*
T0?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:m
gradients/Reshape_grad/ShapeConst*
valueB"   2   *
dtype0*
_output_shapes
:?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
_output_shapes

:2*
T0o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
valueB"   2   *
dtype0?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_2_grad/ShapeConst*
valueB"   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
_output_shapes

:2*
T0o
gradients/Reshape_3_grad/ShapeConst*
valueB"   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
_output_shapes

:2*
T0o
gradients/Reshape_4_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_5_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_6_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_7_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:h
gradients/Reshape_8_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
_output_shapes
:*
T0i
gradients/Reshape_10_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_11_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
_output_shapes
:*
T0i
gradients/Reshape_12_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
_output_shapes
:*
T0i
gradients/Reshape_13_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
_output_shapes
:*
T0i
gradients/Reshape_14_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_15_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
_output_shapes
:*
T0?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
_output_shapes

:2*
T0?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
_output_shapes

:2*
T0?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
_output_shapes

:*
T0?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:??
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
_output_shapes

:2d*
T0?
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes

:d\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: ?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: e
gradients/concat_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_1Const*
valueB:d*
dtype0*
_output_shapes
:?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
_output_shapes
:d*
Index0*
T0?
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:d"G
gradients_concat_grad_slice_1&gradients/concat_grad/Slice_1:output:0"C
gradients_split_grad_concat$gradients/split_grad/concat:output:0"S
#gradients_expanddims_1_grad_reshape,gradients/ExpandDims_1_grad/Reshape:output:0"L
"gradients_transpose_grad_transpose&gradients/transpose_grad/transpose:y:0"G
gradients_split_1_grad_concat&gradients/split_1_grad/concat:output:0"O
!gradients_expanddims_grad_reshape*gradients/ExpandDims_grad/Reshape:output:0*5
forward_function_name__forward_cudnn_lstm_70767*
	_noinline(*?
_input_shapes?
?:?????????:?????????0:?????????:?????????: :0?????????:::::?????????::::?????????:::::0?????????2:?????????:?????????:?<::?????????:?????????: ::::::::: : : : *=
api_implements+)lstm_ffcd560d-1075-4e30-88b0-2ea61de6018e*
api_preferred_deviceGPU:& :  : : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% 
?+
?
while_body_75369
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"????   *
dtype0*
_output_shapes
:?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0time:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:??????????
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	??
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	2?n
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0c
addAddMatMul:product:0MatMul_1:product:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2N
TanhTanhsplit:output:2*'
_output_shapes
:?????????2*
T0U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2R
add_1Addmul:z:0	mul_1:z:0*'
_output_shapes
:?????????2*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????2Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemtensorarrayv2_1time	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: E
add_2Addtimeadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
dtype0*
_output_shapes
: *
value	B :S
add_3Addwhile_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: ?
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0?

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2?

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0" 
strided_slicestrided_slice_0"!

identity_1Identity_1:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : :	 :
 :  : : : : : : 
??
?
!__inference__traced_restore_77416
file_prefix*
&assignvariableop_tech_dense_0_4_kernel*
&assignvariableop_1_tech_dense_0_4_bias,
(assignvariableop_2_tech_dense_3_4_kernel*
&assignvariableop_3_tech_dense_3_4_bias-
)assignvariableop_4_dense_pooling_5_kernel+
'assignvariableop_5_dense_pooling_5_bias/
+assignvariableop_6_dense_pooling_1_4_kernel-
)assignvariableop_7_dense_pooling_1_4_bias)
%assignvariableop_8_dense_out_4_kernel'
#assignvariableop_9_dense_out_4_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate'
#assignvariableop_15_lstm_0_4_kernel1
-assignvariableop_16_lstm_0_4_recurrent_kernel%
!assignvariableop_17_lstm_0_4_bias'
#assignvariableop_18_lstm_1_4_kernel1
-assignvariableop_19_lstm_1_4_recurrent_kernel%
!assignvariableop_20_lstm_1_4_bias
assignvariableop_21_total
assignvariableop_22_count4
0assignvariableop_23_adam_tech_dense_0_4_kernel_m2
.assignvariableop_24_adam_tech_dense_0_4_bias_m4
0assignvariableop_25_adam_tech_dense_3_4_kernel_m2
.assignvariableop_26_adam_tech_dense_3_4_bias_m5
1assignvariableop_27_adam_dense_pooling_5_kernel_m3
/assignvariableop_28_adam_dense_pooling_5_bias_m7
3assignvariableop_29_adam_dense_pooling_1_4_kernel_m5
1assignvariableop_30_adam_dense_pooling_1_4_bias_m1
-assignvariableop_31_adam_dense_out_4_kernel_m/
+assignvariableop_32_adam_dense_out_4_bias_m.
*assignvariableop_33_adam_lstm_0_4_kernel_m8
4assignvariableop_34_adam_lstm_0_4_recurrent_kernel_m,
(assignvariableop_35_adam_lstm_0_4_bias_m.
*assignvariableop_36_adam_lstm_1_4_kernel_m8
4assignvariableop_37_adam_lstm_1_4_recurrent_kernel_m,
(assignvariableop_38_adam_lstm_1_4_bias_m4
0assignvariableop_39_adam_tech_dense_0_4_kernel_v2
.assignvariableop_40_adam_tech_dense_0_4_bias_v4
0assignvariableop_41_adam_tech_dense_3_4_kernel_v2
.assignvariableop_42_adam_tech_dense_3_4_bias_v5
1assignvariableop_43_adam_dense_pooling_5_kernel_v3
/assignvariableop_44_adam_dense_pooling_5_bias_v7
3assignvariableop_45_adam_dense_pooling_1_4_kernel_v5
1assignvariableop_46_adam_dense_pooling_1_4_bias_v1
-assignvariableop_47_adam_dense_out_4_kernel_v/
+assignvariableop_48_adam_dense_out_4_bias_v.
*assignvariableop_49_adam_lstm_0_4_kernel_v8
4assignvariableop_50_adam_lstm_0_4_recurrent_kernel_v,
(assignvariableop_51_adam_lstm_0_4_bias_v.
*assignvariableop_52_adam_lstm_1_4_kernel_v8
4assignvariableop_53_adam_lstm_1_4_recurrent_kernel_v,
(assignvariableop_54_adam_lstm_1_4_bias_v
identity_56??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:7?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:7*?
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927	L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp&assignvariableop_tech_dense_0_4_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp&assignvariableop_1_tech_dense_0_4_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp(assignvariableop_2_tech_dense_3_4_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp&assignvariableop_3_tech_dense_3_4_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0?
AssignVariableOp_4AssignVariableOp)assignvariableop_4_dense_pooling_5_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0?
AssignVariableOp_5AssignVariableOp'assignvariableop_5_dense_pooling_5_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp+assignvariableop_6_dense_pooling_1_4_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp)assignvariableop_7_dense_pooling_1_4_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_dense_out_4_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_out_4_biasIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0	*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0*
dtype0	*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0?
AssignVariableOp_15AssignVariableOp#assignvariableop_15_lstm_0_4_kernelIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
_output_shapes
:*
T0?
AssignVariableOp_16AssignVariableOp-assignvariableop_16_lstm_0_4_recurrent_kernelIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_lstm_0_4_biasIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_lstm_1_4_kernelIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0?
AssignVariableOp_19AssignVariableOp-assignvariableop_19_lstm_1_4_recurrent_kernelIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp!assignvariableop_20_lstm_1_4_biasIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:{
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0{
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
_output_shapes
:*
T0?
AssignVariableOp_23AssignVariableOp0assignvariableop_23_adam_tech_dense_0_4_kernel_mIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp.assignvariableop_24_adam_tech_dense_0_4_bias_mIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp0assignvariableop_25_adam_tech_dense_3_4_kernel_mIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp.assignvariableop_26_adam_tech_dense_3_4_bias_mIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
_output_shapes
:*
T0?
AssignVariableOp_27AssignVariableOp1assignvariableop_27_adam_dense_pooling_5_kernel_mIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp/assignvariableop_28_adam_dense_pooling_5_bias_mIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp3assignvariableop_29_adam_dense_pooling_1_4_kernel_mIdentity_29:output:0*
_output_shapes
 *
dtype0P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp1assignvariableop_30_adam_dense_pooling_1_4_bias_mIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp-assignvariableop_31_adam_dense_out_4_kernel_mIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_dense_out_4_bias_mIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_lstm_0_4_kernel_mIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
_output_shapes
:*
T0?
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_lstm_0_4_recurrent_kernel_mIdentity_34:output:0*
_output_shapes
 *
dtype0P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_lstm_0_4_bias_mIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_lstm_1_4_kernel_mIdentity_36:output:0*
dtype0*
_output_shapes
 P
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adam_lstm_1_4_recurrent_kernel_mIdentity_37:output:0*
dtype0*
_output_shapes
 P
Identity_38IdentityRestoreV2:tensors:38*
_output_shapes
:*
T0?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_lstm_1_4_bias_mIdentity_38:output:0*
dtype0*
_output_shapes
 P
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp0assignvariableop_39_adam_tech_dense_0_4_kernel_vIdentity_39:output:0*
dtype0*
_output_shapes
 P
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp.assignvariableop_40_adam_tech_dense_0_4_bias_vIdentity_40:output:0*
dtype0*
_output_shapes
 P
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp0assignvariableop_41_adam_tech_dense_3_4_kernel_vIdentity_41:output:0*
dtype0*
_output_shapes
 P
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp.assignvariableop_42_adam_tech_dense_3_4_bias_vIdentity_42:output:0*
dtype0*
_output_shapes
 P
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp1assignvariableop_43_adam_dense_pooling_5_kernel_vIdentity_43:output:0*
dtype0*
_output_shapes
 P
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp/assignvariableop_44_adam_dense_pooling_5_bias_vIdentity_44:output:0*
dtype0*
_output_shapes
 P
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp3assignvariableop_45_adam_dense_pooling_1_4_kernel_vIdentity_45:output:0*
dtype0*
_output_shapes
 P
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp1assignvariableop_46_adam_dense_pooling_1_4_bias_vIdentity_46:output:0*
dtype0*
_output_shapes
 P
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp-assignvariableop_47_adam_dense_out_4_kernel_vIdentity_47:output:0*
dtype0*
_output_shapes
 P
Identity_48IdentityRestoreV2:tensors:48*
_output_shapes
:*
T0?
AssignVariableOp_48AssignVariableOp+assignvariableop_48_adam_dense_out_4_bias_vIdentity_48:output:0*
dtype0*
_output_shapes
 P
Identity_49IdentityRestoreV2:tensors:49*
_output_shapes
:*
T0?
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_lstm_0_4_kernel_vIdentity_49:output:0*
dtype0*
_output_shapes
 P
Identity_50IdentityRestoreV2:tensors:50*
_output_shapes
:*
T0?
AssignVariableOp_50AssignVariableOp4assignvariableop_50_adam_lstm_0_4_recurrent_kernel_vIdentity_50:output:0*
dtype0*
_output_shapes
 P
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_lstm_0_4_bias_vIdentity_51:output:0*
dtype0*
_output_shapes
 P
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adam_lstm_1_4_kernel_vIdentity_52:output:0*
dtype0*
_output_shapes
 P
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp4assignvariableop_53_adam_lstm_1_4_recurrent_kernel_vIdentity_53:output:0*
dtype0*
_output_shapes
 P
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_lstm_1_4_bias_vIdentity_54:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B ?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?

Identity_55Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
_output_shapes
: *
T0?

Identity_56IdentityIdentity_55:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_56Identity_56:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::2
RestoreV2_1RestoreV2_12(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_31AssignVariableOp_312$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_49AssignVariableOp_492*
AssignVariableOp_54AssignVariableOp_54:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 :7 
?
b
F__inference_tech_relu_3_layer_call_and_return_conditional_losses_76748

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:?????????Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?P
?
__forward_cudnn_lstm_73143

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*
T0l
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0;

zeros_likeConst*
valueBd*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
	num_split*
T0>
Const_3Const*
valueB:
?????????*
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
No
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
Index0I
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
T0*
Index0C
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"
cudnnrnnCudnnRNN:output:0"-
transpose_5_permtranspose_5/perm:output:0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"%
expanddims_1ExpandDims_1:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0"
concat_1concat_1_0:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_9_permtranspose_9/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_3Identity_3:output:0"-
transpose_8_permtranspose_8/perm:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"-
transpose_2_permtranspose_2/perm:output:0"3
strided_slice_stackstrided_slice/stack:output:0"'
concat_1_axisconcat_1/axis:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"
	transposetranspose_0:y:0"-
transpose_7_permtranspose_7/perm:output:0"+
split_split_dimsplit/split_dim:output:0"#
concat_axisconcat/axis:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"
init_hinit_h_0"-
transpose_6_permtranspose_6/perm:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0*I
backward_function_name/-__inference___backward_cudnn_lstm_73141_73317*
_input_shapes *=
api_implements+)lstm_3f27c04b-652c-4bbb-81f8-497acfa4408c*
api_preferred_deviceGPU*
	_noinline(2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?
?
A__inference_lstm_1_layer_call_and_return_conditional_losses_74730

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
dtype0*
_output_shapes
: *
value
B :?Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
_output_shapes
: *
T0P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:?????????*
T0O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :?*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:??????????
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5**
config_proto

GPU 

CPU2J 8*
Tin

2*o
_output_shapes]
[:?????????:??????????????????:?????????:?????????: *,
_gradient_op_typePartitionedCall-74441*(
f#R!
__inference_standard_lstm_74440*
Tout	
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
?+
?
while_body_71977
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"????   *
dtype0*
_output_shapes
:?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0time:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:??????????
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	??
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	2?n
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0c
addAddMatMul:product:0MatMul_1:product:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:?????????2*
T0S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????2U
mul_1MulSigmoid:y:0Tanh:y:0*'
_output_shapes
:?????????2*
T0R
add_1Addmul:z:0	mul_1:z:0*'
_output_shapes
:?????????2*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2K
Tanh_1Tanh	add_1:z:0*'
_output_shapes
:?????????2*
T0Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*'
_output_shapes
:?????????2*
T0?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemtensorarrayv2_1time	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
_output_shapes
: *
value	B :*
dtype0E
add_2Addtimeadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: S
add_3Addwhile_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: ?
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0?

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:?????????2*
T0?

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:?????????2*
T0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0" 
strided_slicestrided_slice_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: : : : : :	 :
 :  : : : 
?>
?
__inference_cudnn_lstm_75585

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*
T0l
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueB?*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
dtype0*
value	B : W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
?????????*
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
dtype0*
valueB"       L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
Nk
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
Index0*
T0C
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
dtype0*
valueB:?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
	_noinline(*
_input_shapes *=
api_implements+)lstm_7f56906d-3660-42c1-afeb-29d9e0ca9efc*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel
?>
?
__inference_cudnn_lstm_72191

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*
T0l
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueB?*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
dtype0*
valueB:
?????????E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
dtype0*
valueB"       L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0k
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*
	_noinline(*
_input_shapes *=
api_implements+)lstm_6b86d02e-a5e5-4d08-8f29-1ea05985005f*
api_preferred_deviceGPU20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs
?
?
while_cond_72452
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
G
LessLesstimeless_strided_slice*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : ::::  : : : : : : : : :	 :
 
?
?
G__inference_tech_dense_0_layer_call_and_return_conditional_losses_74752

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:2i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:2v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
A__inference_lstm_0_layer_call_and_return_conditional_losses_75297

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :2*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
dtype0*
_output_shapes
: *
value
B :?Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
_output_shapes
: *
T0P
zeros/packed/1Const*
_output_shapes
: *
value	B :2*
dtype0s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
_output_shapes
:*
T0P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:?????????2*
T0O
zeros_1/mul/yConst*
value	B :2*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :?*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
value	B :2*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2?
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*,
_gradient_op_typePartitionedCall-75008*(
f#R!
__inference_standard_lstm_75007*
Tout	
2**
config_proto

GPU 

CPU2J 8*
Tin

2*f
_output_shapesT
R:?????????2:?????????02:?????????2:?????????2: ?
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????02"
identityIdentity:output:0*6
_input_shapes%
#:?????????0:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
?
?
A__inference_lstm_0_layer_call_and_return_conditional_losses_71435

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0M
zeros/mul/yConst*
value	B :2*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
value
B :?*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
_output_shapes
: *
T0P
zeros/packed/1Const*
value	B :2*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
_output_shapes
:*
T0*
NP
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:?????????2*
T0O
zeros_1/mul/yConst*
value	B :2*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :?*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
value	B :2*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2?
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*,
_gradient_op_typePartitionedCall-71146*(
f#R!
__inference_standard_lstm_71145*
Tout	
2**
config_proto

GPU 

CPU2J 8*
Tin

2*o
_output_shapes]
[:?????????2:??????????????????2:?????????2:?????????2: ?
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
?G
?
__inference_standard_lstm_75963

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?whilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:0?????????2B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
valueB :
?????????*
dtype0*
_output_shapes
: ?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: ?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"????2   *
dtype0*
_output_shapes
:?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*'
_output_shapes
:?????????2*
Index0*
T0*
shrink_axis_mask?
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:2d{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????db
addAddMatMul:product:0MatMul_1:product:0*'
_output_shapes
:?????????d*
T0?
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????S
mulMulSigmoid_1:y:0init_c*'
_output_shapes
:?????????*
T0N
TanhTanhsplit:output:2*'
_output_shapes
:?????????*
T0U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????R
add_1Addmul:z:0	mul_1:z:0*'
_output_shapes
:?????????*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*'
_output_shapes
:?????????*
T0h
TensorArrayV2_1/element_shapeConst*
valueB :
?????????*
dtype0*
_output_shapes
: ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
_output_shapes
: *
value	B :0*
dtype0T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
condR
while_cond_75857*
bodyR
while_body_75858*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *
T
2*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
_lower_using_switch_merge(*
parallel_iterations K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:?????????*
T0^
while/Identity_5Identitywhile:output:5*'
_output_shapes
:?????????*
T0M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"????   *
dtype0*
_output_shapes
:?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:0?????????h
strided_slice_2/stackConst*
_output_shapes
:*
valueB:
?????????*
dtype0a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*'
_output_shapes
:?????????*
Index0*
T0*
shrink_axis_maske
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????0[
runtimeConst"/device:CPU:0*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:?????????*
T0?

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:?????????0?

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:??????????

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:??????????

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*=
api_implements+)lstm_07d2f4a3-a6ab-4ba4-8910-49922c38a5c9*
api_preferred_deviceCPU*
	_noinline(*\
_input_shapesK
I:?????????02:?????????:?????????:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs
?v
?
 __inference__wrapped_model_70967

lstm_input

tech_input8
4model_14_tech_dense_0_matmul_readvariableop_resource9
5model_14_tech_dense_0_biasadd_readvariableop_resource8
4model_14_tech_dense_3_matmul_readvariableop_resource9
5model_14_tech_dense_3_biasadd_readvariableop_resource2
.model_14_lstm_0_statefulpartitionedcall_args_32
.model_14_lstm_0_statefulpartitionedcall_args_42
.model_14_lstm_0_statefulpartitionedcall_args_52
.model_14_lstm_1_statefulpartitionedcall_args_32
.model_14_lstm_1_statefulpartitionedcall_args_42
.model_14_lstm_1_statefulpartitionedcall_args_59
5model_14_dense_pooling_matmul_readvariableop_resource:
6model_14_dense_pooling_biasadd_readvariableop_resource;
7model_14_dense_pooling_1_matmul_readvariableop_resource<
8model_14_dense_pooling_1_biasadd_readvariableop_resource5
1model_14_dense_out_matmul_readvariableop_resource6
2model_14_dense_out_biasadd_readvariableop_resource
identity??)model_14/dense_out/BiasAdd/ReadVariableOp?(model_14/dense_out/MatMul/ReadVariableOp?-model_14/dense_pooling/BiasAdd/ReadVariableOp?,model_14/dense_pooling/MatMul/ReadVariableOp?/model_14/dense_pooling_1/BiasAdd/ReadVariableOp?.model_14/dense_pooling_1/MatMul/ReadVariableOp?'model_14/lstm_0/StatefulPartitionedCall?'model_14/lstm_1/StatefulPartitionedCall?,model_14/tech_dense_0/BiasAdd/ReadVariableOp?+model_14/tech_dense_0/MatMul/ReadVariableOp?,model_14/tech_dense_3/BiasAdd/ReadVariableOp?+model_14/tech_dense_3/MatMul/ReadVariableOp?
+model_14/tech_dense_0/MatMul/ReadVariableOpReadVariableOp4model_14_tech_dense_0_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:2?
model_14/tech_dense_0/MatMulMatMul
tech_input3model_14/tech_dense_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2?
,model_14/tech_dense_0/BiasAdd/ReadVariableOpReadVariableOp5model_14_tech_dense_0_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:2?
model_14/tech_dense_0/BiasAddBiasAdd&model_14/tech_dense_0/MatMul:product:04model_14/tech_dense_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2{
model_14/tech_relu_0/ReluRelu&model_14/tech_dense_0/BiasAdd:output:0*'
_output_shapes
:?????????2*
T0?
+model_14/tech_dense_3/MatMul/ReadVariableOpReadVariableOp4model_14_tech_dense_3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:2?
model_14/tech_dense_3/MatMulMatMul'model_14/tech_relu_0/Relu:activations:03model_14/tech_dense_3/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
,model_14/tech_dense_3/BiasAdd/ReadVariableOpReadVariableOp5model_14_tech_dense_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
model_14/tech_dense_3/BiasAddBiasAdd&model_14/tech_dense_3/MatMul:product:04model_14/tech_dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????O
model_14/lstm_0/ShapeShape
lstm_input*
_output_shapes
:*
T0m
#model_14/lstm_0/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:o
%model_14/lstm_0/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:o
%model_14/lstm_0/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0?
model_14/lstm_0/strided_sliceStridedSlicemodel_14/lstm_0/Shape:output:0,model_14/lstm_0/strided_slice/stack:output:0.model_14/lstm_0/strided_slice/stack_1:output:0.model_14/lstm_0/strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask]
model_14/lstm_0/zeros/mul/yConst*
_output_shapes
: *
value	B :2*
dtype0?
model_14/lstm_0/zeros/mulMul&model_14/lstm_0/strided_slice:output:0$model_14/lstm_0/zeros/mul/y:output:0*
T0*
_output_shapes
: _
model_14/lstm_0/zeros/Less/yConst*
value
B :?*
dtype0*
_output_shapes
: ?
model_14/lstm_0/zeros/LessLessmodel_14/lstm_0/zeros/mul:z:0%model_14/lstm_0/zeros/Less/y:output:0*
T0*
_output_shapes
: `
model_14/lstm_0/zeros/packed/1Const*
value	B :2*
dtype0*
_output_shapes
: ?
model_14/lstm_0/zeros/packedPack&model_14/lstm_0/strided_slice:output:0'model_14/lstm_0/zeros/packed/1:output:0*
N*
_output_shapes
:*
T0`
model_14/lstm_0/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: ?
model_14/lstm_0/zerosFill%model_14/lstm_0/zeros/packed:output:0$model_14/lstm_0/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2_
model_14/lstm_0/zeros_1/mul/yConst*
value	B :2*
dtype0*
_output_shapes
: ?
model_14/lstm_0/zeros_1/mulMul&model_14/lstm_0/strided_slice:output:0&model_14/lstm_0/zeros_1/mul/y:output:0*
_output_shapes
: *
T0a
model_14/lstm_0/zeros_1/Less/yConst*
_output_shapes
: *
value
B :?*
dtype0?
model_14/lstm_0/zeros_1/LessLessmodel_14/lstm_0/zeros_1/mul:z:0'model_14/lstm_0/zeros_1/Less/y:output:0*
_output_shapes
: *
T0b
 model_14/lstm_0/zeros_1/packed/1Const*
dtype0*
_output_shapes
: *
value	B :2?
model_14/lstm_0/zeros_1/packedPack&model_14/lstm_0/strided_slice:output:0)model_14/lstm_0/zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:b
model_14/lstm_0/zeros_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0?
model_14/lstm_0/zeros_1Fill'model_14/lstm_0/zeros_1/packed:output:0&model_14/lstm_0/zeros_1/Const:output:0*'
_output_shapes
:?????????2*
T0?
'model_14/lstm_0/StatefulPartitionedCallStatefulPartitionedCall
lstm_inputmodel_14/lstm_0/zeros:output:0 model_14/lstm_0/zeros_1:output:0.model_14_lstm_0_statefulpartitionedcall_args_3.model_14_lstm_0_statefulpartitionedcall_args_4.model_14_lstm_0_statefulpartitionedcall_args_5*
Tout	
2**
config_proto

GPU 

CPU2J 8*
Tin

2*f
_output_shapesT
R:?????????2:?????????02:?????????2:?????????2: *,
_gradient_op_typePartitionedCall-70191*(
f#R!
__inference_standard_lstm_70190u
model_14/lstm_1/ShapeShape0model_14/lstm_0/StatefulPartitionedCall:output:1*
_output_shapes
:*
T0m
#model_14/lstm_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:o
%model_14/lstm_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:o
%model_14/lstm_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
model_14/lstm_1/strided_sliceStridedSlicemodel_14/lstm_1/Shape:output:0,model_14/lstm_1/strided_slice/stack:output:0.model_14/lstm_1/strided_slice/stack_1:output:0.model_14/lstm_1/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: ]
model_14/lstm_1/zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: ?
model_14/lstm_1/zeros/mulMul&model_14/lstm_1/strided_slice:output:0$model_14/lstm_1/zeros/mul/y:output:0*
T0*
_output_shapes
: _
model_14/lstm_1/zeros/Less/yConst*
value
B :?*
dtype0*
_output_shapes
: ?
model_14/lstm_1/zeros/LessLessmodel_14/lstm_1/zeros/mul:z:0%model_14/lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: `
model_14/lstm_1/zeros/packed/1Const*
_output_shapes
: *
value	B :*
dtype0?
model_14/lstm_1/zeros/packedPack&model_14/lstm_1/strided_slice:output:0'model_14/lstm_1/zeros/packed/1:output:0*
_output_shapes
:*
T0*
N`
model_14/lstm_1/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0?
model_14/lstm_1/zerosFill%model_14/lstm_1/zeros/packed:output:0$model_14/lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????_
model_14/lstm_1/zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: ?
model_14/lstm_1/zeros_1/mulMul&model_14/lstm_1/strided_slice:output:0&model_14/lstm_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: a
model_14/lstm_1/zeros_1/Less/yConst*
value
B :?*
dtype0*
_output_shapes
: ?
model_14/lstm_1/zeros_1/LessLessmodel_14/lstm_1/zeros_1/mul:z:0'model_14/lstm_1/zeros_1/Less/y:output:0*
_output_shapes
: *
T0b
 model_14/lstm_1/zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: ?
model_14/lstm_1/zeros_1/packedPack&model_14/lstm_1/strided_slice:output:0)model_14/lstm_1/zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:b
model_14/lstm_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: ?
model_14/lstm_1/zeros_1Fill'model_14/lstm_1/zeros_1/packed:output:0&model_14/lstm_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:??????????
'model_14/lstm_1/StatefulPartitionedCallStatefulPartitionedCall0model_14/lstm_0/StatefulPartitionedCall:output:1model_14/lstm_1/zeros:output:0 model_14/lstm_1/zeros_1:output:0.model_14_lstm_1_statefulpartitionedcall_args_3.model_14_lstm_1_statefulpartitionedcall_args_4.model_14_lstm_1_statefulpartitionedcall_args_5*,
_gradient_op_typePartitionedCall-70654*(
f#R!
__inference_standard_lstm_70653*
Tout	
2**
config_proto

GPU 

CPU2J 8*
Tin

2*f
_output_shapesT
R:?????????:?????????0:?????????:?????????: {
model_14/tech_relu_3/ReluRelu&model_14/tech_dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
 model_14/concatenate/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: ?
model_14/concatenate/concatConcatV20model_14/lstm_1/StatefulPartitionedCall:output:0'model_14/tech_relu_3/Relu:activations:0)model_14/concatenate/concat/axis:output:0*'
_output_shapes
:?????????2*
T0*
N?
,model_14/dense_pooling/MatMul/ReadVariableOpReadVariableOp5model_14_dense_pooling_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:2?
model_14/dense_pooling/MatMulMatMul$model_14/concatenate/concat:output:04model_14/dense_pooling/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
-model_14/dense_pooling/BiasAdd/ReadVariableOpReadVariableOp6model_14_dense_pooling_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
model_14/dense_pooling/BiasAddBiasAdd'model_14/dense_pooling/MatMul:product:05model_14/dense_pooling/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0~
model_14/dense_pooling/ReluRelu'model_14/dense_pooling/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
.model_14/dense_pooling_1/MatMul/ReadVariableOpReadVariableOp7model_14_dense_pooling_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
?
model_14/dense_pooling_1/MatMulMatMul)model_14/dense_pooling/Relu:activations:06model_14/dense_pooling_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
/model_14/dense_pooling_1/BiasAdd/ReadVariableOpReadVariableOp8model_14_dense_pooling_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
?
 model_14/dense_pooling_1/BiasAddBiasAdd)model_14/dense_pooling_1/MatMul:product:07model_14/dense_pooling_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
model_14/dense_pooling_1/ReluRelu)model_14/dense_pooling_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
(model_14/dense_out/MatMul/ReadVariableOpReadVariableOp1model_14_dense_out_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
?
model_14/dense_out/MatMulMatMul+model_14/dense_pooling_1/Relu:activations:00model_14/dense_out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
)model_14/dense_out/BiasAdd/ReadVariableOpReadVariableOp2model_14_dense_out_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
model_14/dense_out/BiasAddBiasAdd#model_14/dense_out/MatMul:product:01model_14/dense_out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
model_14/dense_out/SigmoidSigmoid#model_14/dense_out/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentitymodel_14/dense_out/Sigmoid:y:0*^model_14/dense_out/BiasAdd/ReadVariableOp)^model_14/dense_out/MatMul/ReadVariableOp.^model_14/dense_pooling/BiasAdd/ReadVariableOp-^model_14/dense_pooling/MatMul/ReadVariableOp0^model_14/dense_pooling_1/BiasAdd/ReadVariableOp/^model_14/dense_pooling_1/MatMul/ReadVariableOp(^model_14/lstm_0/StatefulPartitionedCall(^model_14/lstm_1/StatefulPartitionedCall-^model_14/tech_dense_0/BiasAdd/ReadVariableOp,^model_14/tech_dense_0/MatMul/ReadVariableOp-^model_14/tech_dense_3/BiasAdd/ReadVariableOp,^model_14/tech_dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*}
_input_shapesl
j:?????????0:?????????::::::::::::::::2V
)model_14/dense_out/BiasAdd/ReadVariableOp)model_14/dense_out/BiasAdd/ReadVariableOp2^
-model_14/dense_pooling/BiasAdd/ReadVariableOp-model_14/dense_pooling/BiasAdd/ReadVariableOp2`
.model_14/dense_pooling_1/MatMul/ReadVariableOp.model_14/dense_pooling_1/MatMul/ReadVariableOp2R
'model_14/lstm_0/StatefulPartitionedCall'model_14/lstm_0/StatefulPartitionedCall2\
,model_14/dense_pooling/MatMul/ReadVariableOp,model_14/dense_pooling/MatMul/ReadVariableOp2T
(model_14/dense_out/MatMul/ReadVariableOp(model_14/dense_out/MatMul/ReadVariableOp2R
'model_14/lstm_1/StatefulPartitionedCall'model_14/lstm_1/StatefulPartitionedCall2\
,model_14/tech_dense_0/BiasAdd/ReadVariableOp,model_14/tech_dense_0/BiasAdd/ReadVariableOp2b
/model_14/dense_pooling_1/BiasAdd/ReadVariableOp/model_14/dense_pooling_1/BiasAdd/ReadVariableOp2Z
+model_14/tech_dense_3/MatMul/ReadVariableOp+model_14/tech_dense_3/MatMul/ReadVariableOp2\
,model_14/tech_dense_3/BiasAdd/ReadVariableOp,model_14/tech_dense_3/BiasAdd/ReadVariableOp2Z
+model_14/tech_dense_0/MatMul/ReadVariableOp+model_14/tech_dense_0/MatMul/ReadVariableOp: : : : : : :	 :
 : : : : : : : :* &
$
_user_specified_name
lstm_input:*&
$
_user_specified_name
tech_input: 
?>
?
__inference_cudnn_lstm_75118

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
dtype0*
value	B :;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueB?*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
?????????*
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
Nk
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
dtype0*
valueB: C
strided_slice/stack_2Const*
dtype0*
valueB:?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
dtype0*
valueB:E
strided_slice_1/stack_2Const*
dtype0*
valueB:?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
strided_slice_2/stackConst*
dtype0*
valueB: E
strided_slice_2/stack_1Const*
dtype0*
valueB:E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*
	_noinline(*
_input_shapes *=
api_implements+)lstm_68d0f6f7-76c5-4200-8d0a-18146dc4d807*
api_preferred_deviceGPU2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c
?
?
A__inference_lstm_0_layer_call_and_return_conditional_losses_72846

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
dtype0*
_output_shapes
: *
value	B :2_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
value
B :?*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :2*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2O
zeros_1/mul/yConst*
value	B :2*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
_output_shapes
: *
value
B :?*
dtype0_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
dtype0*
_output_shapes
: *
value	B :2w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*'
_output_shapes
:?????????2*
T0?
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5**
config_proto

GPU 

CPU2J 8*
Tin

2*o
_output_shapes]
[:?????????2:??????????????????2:?????????2:?????????2: *,
_gradient_op_typePartitionedCall-72557*(
f#R!
__inference_standard_lstm_72556*
Tout	
2?
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
?
?
&__inference_lstm_0_layer_call_fn_75773

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*,
_gradient_op_typePartitionedCall-75767*J
fERC
A__inference_lstm_0_layer_call_and_return_conditional_losses_75297*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:?????????02?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????02"
identityIdentity:output:0*6
_input_shapes%
#:?????????0:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
?
?
while_cond_71041
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
G
LessLesstimeless_strided_slice*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: : : :	 :
 :  : : : : : 
?
?
&__inference_lstm_0_layer_call_fn_72377

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*,
_gradient_op_typePartitionedCall-72371*J
fERC
A__inference_lstm_0_layer_call_and_return_conditional_losses_72370*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*4
_output_shapes"
 :??????????????????2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*4
_output_shapes"
 :??????????????????2*
T0"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
?
a
E__inference_lstm_input_layer_call_and_return_conditional_losses_74791

inputs
identityR
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????0"
identityIdentity:output:0**
_input_shapes
:?????????0:& "
 
_user_specified_nameinputs
?
?
while_cond_74901
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
less_1_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
[
LessLesswhile_loop_counterwhile_maximum_iterations*
_output_shapes
: *
T0K
Less_1Lesstimeless_1_strided_slice*
_output_shapes
: *
T0F

LogicalAnd
LogicalAndLess:z:0
Less_1:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: : : : :	 :
 :  : : : : 
?>
?
__inference_cudnn_lstm_71256

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*
dtype0*!
valueB"          @
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
dtype0*
value	B : F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*
T01
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueB?*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
dtype0*
value	B : W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
?????????*
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
dtype0*
valueB"       L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
dtype0*
valueB"       L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
dtype0*
valueB"       N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
Nk
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
Index0I
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
dtype0*
valueB:?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
Index0*
T0C
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*
	_noinline(*
_input_shapes *=
api_implements+)lstm_7622bbb0-93a1-4322-8e09-ef1893ca536d*
api_preferred_deviceGPU26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?>
?
__inference_cudnn_lstm_76074

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*
T01
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0;

zeros_likeConst*
valueBd*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
dtype0*
value	B : n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
	num_split*
T0>
Const_3Const*
valueB:
?????????*
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
dtype0*
valueB"       L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
dtype0*
valueB"       N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
Nk
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
dtype0*
valueB:?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*
T0*
Index0C
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapes *=
api_implements+)lstm_07d2f4a3-a6ab-4ba4-8910-49922c38a5c9*
api_preferred_deviceGPU*
	_noinline(26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?G
?
__inference_standard_lstm_70190

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?whilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:0?????????B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: ?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"????   *
dtype0*
_output_shapes
:?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*'
_output_shapes
:?????????*
T0*
Index0*
shrink_axis_mask?
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	2?n
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
addAddMatMul:product:0MatMul_1:product:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2N
TanhTanhsplit:output:2*'
_output_shapes
:?????????2*
T0U
mul_1MulSigmoid:y:0Tanh:y:0*'
_output_shapes
:?????????2*
T0R
add_1Addmul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????2Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*'
_output_shapes
:?????????2*
T0h
TensorArrayV2_1/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
??????????
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
value	B :0T
while/loop_counterConst*
dtype0*
_output_shapes
: *
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_70084*
bodyR
while_body_70085*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *
T
2*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:?????????2^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:?????????2M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"????2   *
dtype0*
_output_shapes
:?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:0?????????2h
strided_slice_2/stackConst*
valueB:
?????????*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:?????????2*
Index0*
T0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*+
_output_shapes
:?????????02*
T0[
runtimeConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ???
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2?

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*+
_output_shapes
:?????????02*
T0?

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2?

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:?????????2*
T0?

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
	_noinline(*\
_input_shapesK
I:?????????0:?????????2:?????????2:::*=
api_implements+)lstm_0e2cd939-c429-480a-8893-82e6e02dfe56*
api_preferred_deviceCPU22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel
?
F
*__inference_lstm_input_layer_call_fn_74800

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tout
2**
config_proto

GPU 

CPU2J 8*+
_output_shapes
:?????????0*
Tin
2*,
_gradient_op_typePartitionedCall-74797*N
fIRG
E__inference_lstm_input_layer_call_and_return_conditional_losses_74791d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????0"
identityIdentity:output:0**
_input_shapes
:?????????0:& "
 
_user_specified_nameinputs
??
?
-__inference___backward_cudnn_lstm_75586_75762
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis&
"gradients_transpose_grad_transpose%
!gradients_expanddims_grad_reshape'
#gradients_expanddims_1_grad_reshape
gradients_split_grad_concat!
gradients_split_1_grad_concat!
gradients_concat_grad_slice_1?^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:?????????02`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:?????????2O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
shrink_axis_mask*+
_output_shapes
:0?????????2*
Index0*
T0?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*+
_output_shapes
:0?????????2*
T0?
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*+
_output_shapes
:?????????2*
T0*
Index0*
shrink_axis_mask?
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
_output_shapes
:*
T0?
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*
shrink_axis_mask*+
_output_shapes
:?????????2*
Index0*
T0?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*+
_output_shapes
:0?????????2c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
_output_shapes
:*
T0?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*`
_output_shapesN
L:0?????????:?????????2:?????????2:?R*
T0?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*+
_output_shapes
:?????????0*
T0u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: ?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: g
gradients/concat_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:2i
gradients/concat_1_grad/Shape_1Const*
valueB:2*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_2Const*
valueB:2*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_3Const*
dtype0*
_output_shapes
:*
valueB:2j
gradients/concat_1_grad/Shape_4Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
valueB:?*
dtype0j
gradients/concat_1_grad/Shape_6Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_7Const*
valueB:?*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_8Const*
valueB:2*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
dtype0*
_output_shapes
:*
valueB:2j
 gradients/concat_1_grad/Shape_10Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_11Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_12Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_13Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_14Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_15Const*
valueB:2*
dtype0*
_output_shapes
:?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:2?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:2?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
_output_shapes
:2*
Index0*
T0?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
_output_shapes
:2*
Index0*
T0?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
_output_shapes
:2*
Index0*
T0?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
_output_shapes
:2*
Index0*
T0?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
_output_shapes
:2*
Index0*
T0?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
_output_shapes
:2*
Index0*
T0?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:2m
gradients/Reshape_grad/ShapeConst*
valueB"2      *
dtype0*
_output_shapes
:?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
_output_shapes

:2*
T0o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
valueB"2      *
dtype0?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_2_grad/ShapeConst*
valueB"2      *
dtype0*
_output_shapes
:?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_3_grad/ShapeConst*
valueB"2      *
dtype0*
_output_shapes
:?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_4_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:22o
gradients/Reshape_5_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
_output_shapes

:22*
T0o
gradients/Reshape_6_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:22o
gradients/Reshape_7_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:22h
gradients/Reshape_8_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:2h
gradients/Reshape_9_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_10_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:2?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
_output_shapes
:2*
T0i
gradients/Reshape_11_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_12_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
_output_shapes
:2*
T0i
gradients/Reshape_13_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_14_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
_output_shapes
:2*
T0i
gradients/Reshape_15_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:2?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
_output_shapes

:2*
T0?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:22?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:22?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
_output_shapes

:22*
T0?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:22?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:??
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	??
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	2?\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: ?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
valueB:?*
dtype0h
gradients/concat_grad/Shape_1Const*
valueB:?*
dtype0*
_output_shapes
:?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
_output_shapes	
:?*
Index0*
T0"G
gradients_split_1_grad_concat&gradients/split_1_grad/concat:output:0"O
!gradients_expanddims_grad_reshape*gradients/ExpandDims_grad/Reshape:output:0"G
gradients_concat_grad_slice_1&gradients/concat_grad/Slice_1:output:0"C
gradients_split_grad_concat$gradients/split_grad/concat:output:0"S
#gradients_expanddims_1_grad_reshape,gradients/ExpandDims_1_grad/Reshape:output:0"L
"gradients_transpose_grad_transpose&gradients/transpose_grad/transpose:y:0*
	_noinline(*?
_input_shapes?
?:?????????2:?????????02:?????????2:?????????2: :0?????????2:::::?????????2::::?????????2:::::0?????????:?????????2:?????????2:?R::?????????2:?????????2: ::::::::: : : : *=
api_implements+)lstm_7f56906d-3660-42c1-afeb-29d9e0ca9efc*
api_preferred_deviceGPU*5
forward_function_name__forward_cudnn_lstm_75588: : : :  :! :" :# :$ :% :& :  : : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : 
?	
?
H__inference_dense_pooling_layer_call_and_return_conditional_losses_76793

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
??
?
-__inference___backward_cudnn_lstm_72192_72368
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis&
"gradients_transpose_grad_transpose%
!gradients_expanddims_grad_reshape'
#gradients_expanddims_1_grad_reshape
gradients_split_grad_concat!
gradients_split_1_grad_concat!
gradients_concat_grad_slice_1?^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2m
gradients/grad_ys_1Identityplaceholder_1*4
_output_shapes"
 :??????????????????2*
T0`
gradients/grad_ys_2Identityplaceholder_2*'
_output_shapes
:?????????2*
T0`
gradients/grad_ys_3Identityplaceholder_3*'
_output_shapes
:?????????2*
T0O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
shrink_axis_mask*4
_output_shapes"
 :??????????????????2*
Index0*
T0?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*4
_output_shapes"
 :??????????????????2*
T0?
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
_output_shapes
:*
T0?
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
T0*
Index0*
shrink_axis_mask*+
_output_shapes
:?????????2?
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
_output_shapes
:*
T0?
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:?????????2?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*4
_output_shapes"
 :??????????????????2*
T0c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
_output_shapes
:*
T0?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*i
_output_shapesW
U:??????????????????:?????????2:?????????2:?R*
T0?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*4
_output_shapes"
 :??????????????????*
T0u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
_output_shapes
:*
T0?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*'
_output_shapes
:?????????2*
T0^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: ?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
_output_shapes
: *
T0g
gradients/concat_1_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_1Const*
valueB:2*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_2Const*
dtype0*
_output_shapes
:*
valueB:2i
gradients/concat_1_grad/Shape_3Const*
valueB:2*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_4Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
valueB:?*
dtype0j
gradients/concat_1_grad/Shape_6Const*
dtype0*
_output_shapes
:*
valueB:?j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
valueB:?*
dtype0i
gradients/concat_1_grad/Shape_8Const*
valueB:2*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_11Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_12Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_13Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_14Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_15Const*
valueB:2*
dtype0*
_output_shapes
:?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:2?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:2?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes
:2?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes
:2?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:2?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
_output_shapes
:2*
Index0*
T0m
gradients/Reshape_grad/ShapeConst*
valueB"2      *
dtype0*
_output_shapes
:?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_1_grad/ShapeConst*
valueB"2      *
dtype0*
_output_shapes
:?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_2_grad/ShapeConst*
valueB"2      *
dtype0*
_output_shapes
:?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_3_grad/ShapeConst*
valueB"2      *
dtype0*
_output_shapes
:?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_4_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"2   2   ?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:22o
gradients/Reshape_5_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:22o
gradients/Reshape_6_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:22o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
valueB"2   2   *
dtype0?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
_output_shapes

:22*
T0h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
valueB:2*
dtype0?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:2h
gradients/Reshape_9_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_10_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_11_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_12_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
_output_shapes
:2*
T0i
gradients/Reshape_13_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_14_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
_output_shapes
:2*
T0i
gradients/Reshape_15_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
_output_shapes
:2*
T0?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
_output_shapes

:2*
T0?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
_output_shapes

:2*
T0?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
_output_shapes

:2*
T0?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:22?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
_output_shapes

:22*
T0?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:22?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
_output_shapes

:22*
T0?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
_output_shapes	
:?*
T0?
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	??
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	2?\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: ?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:?*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:??
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
_output_shapes	
:?*
Index0*
T0"G
gradients_split_1_grad_concat&gradients/split_1_grad/concat:output:0"O
!gradients_expanddims_grad_reshape*gradients/ExpandDims_grad/Reshape:output:0"G
gradients_concat_grad_slice_1&gradients/concat_grad/Slice_1:output:0"C
gradients_split_grad_concat$gradients/split_grad/concat:output:0"S
#gradients_expanddims_1_grad_reshape,gradients/ExpandDims_1_grad/Reshape:output:0"L
"gradients_transpose_grad_transpose&gradients/transpose_grad/transpose:y:0*5
forward_function_name__forward_cudnn_lstm_72194*
	_noinline(*?
_input_shapes?
?:?????????2:??????????????????2:?????????2:?????????2: :??????????????????2:::::?????????2::::?????????2:::::??????????????????:?????????2:?????????2:?R::?????????2:?????????2: ::::::::: : : : *=
api_implements+)lstm_6b86d02e-a5e5-4d08-8f29-1ea05985005f*
api_preferred_deviceGPU:  : : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& 
??
?
-__inference___backward_cudnn_lstm_74552_74728
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis&
"gradients_transpose_grad_transpose%
!gradients_expanddims_grad_reshape'
#gradients_expanddims_1_grad_reshape
gradients_split_grad_concat!
gradients_split_1_grad_concat!
gradients_concat_grad_slice_1?^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????m
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :??????????????????`
gradients/grad_ys_2Identityplaceholder_2*'
_output_shapes
:?????????*
T0`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:?????????O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*4
_output_shapes"
 :??????????????????*
T0*
Index0*
shrink_axis_mask?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :???????????????????
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*+
_output_shapes
:?????????*
Index0*
T0*
shrink_axis_mask?
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*
T0*
Index0*
shrink_axis_mask*+
_output_shapes
:??????????
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*4
_output_shapes"
 :??????????????????c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
_output_shapes
:*
T0?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*i
_output_shapesW
U:??????????????????2:?????????:?????????:?<*
T0?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :??????????????????2u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
_output_shapes
:*
T0?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*'
_output_shapes
:?????????*
T0^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: ?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
valueB:?	*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_1Const*
valueB:?	*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_2Const*
valueB:?	*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_3Const*
valueB:?	*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_4Const*
dtype0*
_output_shapes
:*
valueB:?j
gradients/concat_1_grad/Shape_5Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_6Const*
dtype0*
_output_shapes
:*
valueB:?j
gradients/concat_1_grad/Shape_7Const*
valueB:?*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_8Const*
valueB:*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_11Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_12Const*
dtype0*
_output_shapes
:*
valueB:j
 gradients/concat_1_grad/Shape_13Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_14Const*
dtype0*
_output_shapes
:*
valueB:j
 gradients/concat_1_grad/Shape_15Const*
dtype0*
_output_shapes
:*
valueB:?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:?	?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:?	?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:?	?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:?	?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
_output_shapes
:*
Index0*
T0?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
_output_shapes
:*
Index0*
T0?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
_output_shapes
:*
Index0*
T0?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:m
gradients/Reshape_grad/ShapeConst*
valueB"   2   *
dtype0*
_output_shapes
:?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_1_grad/ShapeConst*
valueB"   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_2_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"   2   ?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_3_grad/ShapeConst*
valueB"   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
valueB"      *
dtype0?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
valueB"      *
dtype0?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
valueB"      *
dtype0?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
valueB"      *
dtype0?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:h
gradients/Reshape_8_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:h
gradients/Reshape_9_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_10_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
_output_shapes
:*
T0i
gradients/Reshape_11_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_12_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_14_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
_output_shapes
:*
T0i
gradients/Reshape_15_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
_output_shapes

:2*
T0?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:??
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
_output_shapes

:2d*
T0?
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes

:d\
gradients/concat_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
_output_shapes
: *
T0e
gradients/concat_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_1Const*
valueB:d*
dtype0*
_output_shapes
:?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:d?
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:d"G
gradients_concat_grad_slice_1&gradients/concat_grad/Slice_1:output:0"C
gradients_split_grad_concat$gradients/split_grad/concat:output:0"S
#gradients_expanddims_1_grad_reshape,gradients/ExpandDims_1_grad/Reshape:output:0"L
"gradients_transpose_grad_transpose&gradients/transpose_grad/transpose:y:0"G
gradients_split_1_grad_concat&gradients/split_1_grad/concat:output:0"O
!gradients_expanddims_grad_reshape*gradients/ExpandDims_grad/Reshape:output:0*5
forward_function_name__forward_cudnn_lstm_74554*
	_noinline(*?
_input_shapes?
?:?????????:??????????????????:?????????:?????????: :??????????????????:::::?????????::::?????????:::::??????????????????2:?????????:?????????:?<::?????????:?????????: ::::::::: : : : *=
api_implements+)lstm_583a098d-9984-4e53-9565-7f3e396fc7ec*
api_preferred_deviceGPU:& :  : : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% 
?P
?
__forward_cudnn_lstm_72670

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueB?*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
dtype0*
value	B : n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
?????????*
dtype0E
transpose_1/permConst*
dtype0*
valueB"       L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
dtype0*
valueB"       L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
dtype0*
valueB"       N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
dtype0*
valueB"       N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
No
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0")
transpose_permtranspose/perm:output:0"!

expanddimsExpandDims:output:0"-
transpose_4_permtranspose_4/perm:output:0"%
expanddims_1ExpandDims_1:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0"
concat_1concat_1_0:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_9_permtranspose_9/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"!

identity_1Identity_1:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"-
transpose_8_permtranspose_8/perm:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"3
strided_slice_stackstrided_slice/stack:output:0"-
transpose_2_permtranspose_2/perm:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"'
concat_1_axisconcat_1/axis:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"-
transpose_7_permtranspose_7/perm:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"-
transpose_5_permtranspose_5/perm:output:0*
_input_shapes *=
api_implements+)lstm_a73b20f0-c393-4dea-a4aa-de0c2ac2517c*
api_preferred_deviceGPU*
	_noinline(*I
backward_function_name/-__inference___backward_cudnn_lstm_72668_7284426
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?P
?
__forward_cudnn_lstm_70304

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
dtype0*
value	B : D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*
T01
Const_1Const*
dtype0*
value	B :;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueB?*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
?????????*
dtype0E
transpose_1/permConst*
dtype0*
valueB"       L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
dtype0*
valueB"       L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
dtype0*
valueB"       L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
No
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"'
concat_1_axisconcat_1/axis:output:0"
	transposetranspose_0:y:0"-
transpose_7_permtranspose_7/perm:output:0"+
split_split_dimsplit/split_dim:output:0"#
concat_axisconcat/axis:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"-
transpose_5_permtranspose_5/perm:output:0")
transpose_permtranspose/perm:output:0"!

expanddimsExpandDims:output:0"-
transpose_4_permtranspose_4/perm:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"%
expanddims_1ExpandDims_1:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0"
concat_1concat_1_0:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_9_permtranspose_9/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"-
transpose_3_permtranspose_3/perm:output:0"!

identity_1Identity_1:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"-
transpose_8_permtranspose_8/perm:output:0"
identityIdentity:output:0"-
transpose_2_permtranspose_2/perm:output:0"3
strided_slice_stackstrided_slice/stack:output:0*I
backward_function_name/-__inference___backward_cudnn_lstm_70302_70478*
_input_shapes *=
api_implements+)lstm_0e2cd939-c429-480a-8893-82e6e02dfe56*
api_preferred_deviceGPU*
	_noinline(26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
??
?
-__inference___backward_cudnn_lstm_76542_76718
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis&
"gradients_transpose_grad_transpose%
!gradients_expanddims_grad_reshape'
#gradients_expanddims_1_grad_reshape
gradients_split_grad_concat!
gradients_split_1_grad_concat!
gradients_concat_grad_slice_1?^
gradients/grad_ys_0Identityplaceholder*'
_output_shapes
:?????????*
T0d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:?????????0`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????`
gradients/grad_ys_3Identityplaceholder_3*'
_output_shapes
:?????????*
T0O
gradients/grad_ys_4Identityplaceholder_4*
_output_shapes
: *
T0}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
_output_shapes
:*
T0?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*+
_output_shapes
:0?????????*
Index0*
T0*
shrink_axis_mask?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*+
_output_shapes
:0?????????*
T0?
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
_output_shapes
:*
T0?
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*+
_output_shapes
:?????????*
Index0*
T0*
shrink_axis_mask?
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*+
_output_shapes
:?????????*
T0*
Index0*
shrink_axis_mask?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*+
_output_shapes
:0?????????c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*`
_output_shapesN
L:0?????????2:?????????:?????????:?<*
T0?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:?????????02u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
_output_shapes
:*
T0?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*'
_output_shapes
:?????????*
T0^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: ?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
valueB:?	*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_1Const*
valueB:?	*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_2Const*
dtype0*
_output_shapes
:*
valueB:?	j
gradients/concat_1_grad/Shape_3Const*
valueB:?	*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_4Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_5Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_6Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_7Const*
valueB:?*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_8Const*
valueB:*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_11Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
valueB:*
dtype0j
 gradients/concat_1_grad/Shape_13Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_14Const*
dtype0*
_output_shapes
:*
valueB:j
 gradients/concat_1_grad/Shape_15Const*
valueB:*
dtype0*
_output_shapes
:?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:?	?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:?	?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:?	?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
_output_shapes	
:?	*
Index0*
T0?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
_output_shapes
:*
Index0*
T0?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
_output_shapes
:*
Index0*
T0?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
_output_shapes
:*
Index0*
T0?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
_output_shapes
:*
Index0*
T0?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
valueB"   2   *
dtype0?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
_output_shapes

:2*
T0o
gradients/Reshape_1_grad/ShapeConst*
valueB"   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_2_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"   2   ?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_3_grad/ShapeConst*
valueB"   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_4_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_5_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_6_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_7_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:h
gradients/Reshape_8_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:h
gradients/Reshape_9_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_10_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_12_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_13_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_14_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
_output_shapes
:*
T0?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
_output_shapes

:2*
T0?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
_output_shapes

:2*
T0?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
_output_shapes

:*
T0?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
_output_shapes

:*
T0?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:??
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes

:2d?
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes

:d\
gradients/concat_grad/RankConst*
_output_shapes
: *
value	B :*
dtype0?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
_output_shapes
: *
T0e
gradients/concat_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_1Const*
valueB:d*
dtype0*
_output_shapes
:?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:d?
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:d"G
gradients_split_1_grad_concat&gradients/split_1_grad/concat:output:0"O
!gradients_expanddims_grad_reshape*gradients/ExpandDims_grad/Reshape:output:0"G
gradients_concat_grad_slice_1&gradients/concat_grad/Slice_1:output:0"C
gradients_split_grad_concat$gradients/split_grad/concat:output:0"S
#gradients_expanddims_1_grad_reshape,gradients/ExpandDims_1_grad/Reshape:output:0"L
"gradients_transpose_grad_transpose&gradients/transpose_grad/transpose:y:0*=
api_implements+)lstm_d99063c4-b3b5-4528-b76f-dadf8da915f1*
api_preferred_deviceGPU*5
forward_function_name__forward_cudnn_lstm_76544*
	_noinline(*?
_input_shapes?
?:?????????:?????????0:?????????:?????????: :0?????????:::::?????????::::?????????:::::0?????????2:?????????:?????????:?<::?????????:?????????: ::::::::: : : : :  : : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& 
?P
?
__forward_cudnn_lstm_72194

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
dtype0*
value	B :9
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueB?*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
	num_split*
T0>
Const_3Const*
dtype0*
valueB:
?????????E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
dtype0*
valueB"       L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
dtype0*
valueB"       L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
dtype0*
valueB"       N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
No
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
Index0I
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
Index0*
T0C
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"-
transpose_9_permtranspose_9/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"-
transpose_3_permtranspose_3/perm:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"-
transpose_8_permtranspose_8/perm:output:0"
identityIdentity:output:0"3
strided_slice_stackstrided_slice/stack:output:0"-
transpose_2_permtranspose_2/perm:output:0"'
concat_1_axisconcat_1/axis:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"-
transpose_5_permtranspose_5/perm:output:0")
transpose_permtranspose/perm:output:0"!

expanddimsExpandDims:output:0"-
transpose_4_permtranspose_4/perm:output:0"%
expanddims_1ExpandDims_1:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0"
concat_1concat_1_0:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0*I
backward_function_name/-__inference___backward_cudnn_lstm_72192_72368*
_input_shapes *=
api_implements+)lstm_6b86d02e-a5e5-4d08-8f29-1ea05985005f*
api_preferred_deviceGPU*
	_noinline(26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?
?
&__inference_lstm_0_layer_call_fn_72853

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3**
config_proto

GPU 

CPU2J 8*4
_output_shapes"
 :??????????????????2*
Tin
2*,
_gradient_op_typePartitionedCall-72847*J
fERC
A__inference_lstm_0_layer_call_and_return_conditional_losses_72846*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*4
_output_shapes"
 :??????????????????2*
T0"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
?
?
A__inference_lstm_1_layer_call_and_return_conditional_losses_74254

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :?*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
dtype0*
_output_shapes
: *
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:?????????*
T0O
zeros_1/mul/yConst*
dtype0*
_output_shapes
: *
value	B :c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :?*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
dtype0*
_output_shapes
: *
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:??????????
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tout	
2**
config_proto

GPU 

CPU2J 8*
Tin

2*o
_output_shapes]
[:?????????:??????????????????:?????????:?????????: *,
_gradient_op_typePartitionedCall-73965*(
f#R!
__inference_standard_lstm_73964?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
?H
?
__inference_standard_lstm_74440

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?whilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
?????????*
dtype0*
_output_shapes
: ?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: ?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"????2   *
dtype0*
_output_shapes
:?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:?????????2?
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:2d{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????db
addAddMatMul:product:0MatMul_1:product:0*'
_output_shapes
:?????????d*
T0?
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:?????????:?????????:?????????:?????????T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:?????????*
T0V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????N
TanhTanhsplit:output:2*'
_output_shapes
:?????????*
T0U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????R
add_1Addmul:z:0	mul_1:z:0*'
_output_shapes
:?????????*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*'
_output_shapes
:?????????*
T0h
TensorArrayV2_1/element_shapeConst*
valueB :
?????????*
dtype0*
_output_shapes
: ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
_output_shapes
: *
valueB :
?????????*
dtype0T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
bodyR
while_body_74337*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *K
output_shapes:
8: : : : :?????????:?????????: : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_74336K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:?????????^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:?????????M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"????   *
dtype0*
_output_shapes
:?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :??????????????????h
strided_slice_2/stackConst*
valueB:
?????????*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:?????????*
T0*
Index0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????[
runtimeConst"/device:CPU:0*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:??????????

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :???????????????????

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:??????????

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:??????????

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*e
_input_shapesT
R:??????????????????2:?????????:?????????:::*=
api_implements+)lstm_583a098d-9984-4e53-9565-7f3e396fc7ec*
api_preferred_deviceCPU*
	_noinline(22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs
?
?
&__inference_lstm_0_layer_call_fn_75782

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*,
_gradient_op_typePartitionedCall-75776*J
fERC
A__inference_lstm_0_layer_call_and_return_conditional_losses_75764*
Tout
2**
config_proto

GPU 

CPU2J 8*+
_output_shapes
:?????????02*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*+
_output_shapes
:?????????02*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????0:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
?
?
while_cond_73860
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
G
LessLesstimeless_strided_slice*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::: : : : : : : : :	 :
 :  
?
?
&__inference_lstm_1_layer_call_fn_74261

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*,
_gradient_op_typePartitionedCall-74255*J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_74254*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
?
?
A__inference_lstm_1_layer_call_and_return_conditional_losses_76253

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0M
zeros/mul/yConst*
dtype0*
_output_shapes
: *
value	B :_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :?*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
_output_shapes
:*
T0P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:?????????*
T0O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :?*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*'
_output_shapes
:?????????*
T0?
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*(
f#R!
__inference_standard_lstm_75963*
Tout	
2**
config_proto

GPU 

CPU2J 8*f
_output_shapesT
R:?????????:?????????0:?????????:?????????: *
Tin

2*,
_gradient_op_typePartitionedCall-75964?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*6
_input_shapes%
#:?????????02:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
?
p
F__inference_concatenate_layer_call_and_return_conditional_losses_76768

inputs
inputs_1
identityM
concat/axisConst*
value	B :*
dtype0*
_output_shapes
: u
concatConcatV2inputsinputs_1concat/axis:output:0*
T0*
N*'
_output_shapes
:?????????2W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?P
?
__forward_cudnn_lstm_71259

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
dtype0*
value	B :;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueB?*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
dtype0*
value	B : n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
?????????*
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
dtype0*
valueB"       L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
No
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
dtype0*
valueB: E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"%
expanddims_1ExpandDims_1:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"
concat_1concat_1_0:output:0"-
transpose_9_permtranspose_9/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"-
transpose_3_permtranspose_3/perm:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"-
transpose_8_permtranspose_8/perm:output:0"
identityIdentity:output:0"-
transpose_2_permtranspose_2/perm:output:0"3
strided_slice_stackstrided_slice/stack:output:0"'
concat_1_axisconcat_1/axis:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"-
transpose_5_permtranspose_5/perm:output:0*
	_noinline(*I
backward_function_name/-__inference___backward_cudnn_lstm_71257_71433*
_input_shapes *=
api_implements+)lstm_7622bbb0-93a1-4322-8e09-ef1893ca536d*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?+
?
while_body_70548
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
valueB"????2   *
dtype0?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0time:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:?????????2?
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:2d?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????db
addAddMatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
T0T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:?????????*
T0V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:?????????*
T0S
mulMulSigmoid_1:y:0init_c*'
_output_shapes
:?????????*
T0N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????R
add_1Addmul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????K
Tanh_1Tanh	add_1:z:0*'
_output_shapes
:?????????*
T0Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:??????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemtensorarrayv2_1time	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: E
add_2Addtimeadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: S
add_3Addwhile_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: ?
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0?

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:??????????

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: : : : : :	 :
 :  : : : 
?H
?
__inference_standard_lstm_72556

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?whilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????B
ShapeShapetranspose:y:0*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: ?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"????   *
dtype0*
_output_shapes
:?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:?????????*
T0*
Index0?
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	2?n
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
addAddMatMul:product:0MatMul_1:product:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????G
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????2U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2R
add_1Addmul:z:0	mul_1:z:0*'
_output_shapes
:?????????2*
T0V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:?????????2*
T0K
Tanh_1Tanh	add_1:z:0*'
_output_shapes
:?????????2*
T0Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2h
TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
valueB :
?????????*
dtype0?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
_output_shapes
: *
value	B : *
dtype0c
while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
valueB :
?????????T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
parallel_iterations *
condR
while_cond_72452*
bodyR
while_body_72453*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *
T
2*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
_lower_using_switch_merge(K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:?????????2^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:?????????2M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
valueB"????2   *
dtype0?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :??????????????????2h
strided_slice_2/stackConst*
valueB:
?????????*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:?????????2*
T0*
Index0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*4
_output_shapes"
 :??????????????????2*
T0[
runtimeConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ???
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:?????????2*
T0?

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :??????????????????2?

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2?

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2?

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*e
_input_shapesT
R:??????????????????:?????????2:?????????2:::*=
api_implements+)lstm_a73b20f0-c393-4dea-a4aa-de0c2ac2517c*
api_preferred_deviceCPU*
	_noinline(20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?G
?
__inference_standard_lstm_75007

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?whilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:0?????????B
ShapeShapetranspose:y:0*
_output_shapes
:*
T0]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
valueB :
?????????*
dtype0*
_output_shapes
: ?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: ?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"????   *
dtype0*
_output_shapes
:?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*'
_output_shapes
:?????????*
Index0*
T0*
shrink_axis_mask?
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	2?n
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
addAddMatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:???????????
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2S
mulMulSigmoid_1:y:0init_c*'
_output_shapes
:?????????2*
T0N
TanhTanhsplit:output:2*'
_output_shapes
:?????????2*
T0U
mul_1MulSigmoid:y:0Tanh:y:0*'
_output_shapes
:?????????2*
T0R
add_1Addmul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????2Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2h
TensorArrayV2_1/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
??????????
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
_output_shapes
: *
value	B :0*
dtype0T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_74901*
bodyR
while_body_74902*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
T
2K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
_output_shapes
: *
T0M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:?????????2^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:?????????2M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"????2   *
dtype0*
_output_shapes
:?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:0?????????2h
strided_slice_2/stackConst*
_output_shapes
:*
valueB:
?????????*
dtype0a
strided_slice_2/stack_1Const*
_output_shapes
:*
valueB: *
dtype0a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*'
_output_shapes
:?????????2*
T0*
Index0*
shrink_axis_maske
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????02[
runtimeConst"/device:CPU:0*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:?????????2*
T0?

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:?????????02?

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2?

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:?????????2*
T0?

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*\
_input_shapesK
I:?????????0:?????????2:?????????2:::*=
api_implements+)lstm_68d0f6f7-76c5-4200-8d0a-18146dc4d807*
api_preferred_deviceCPU*
	_noinline(2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
??
?
-__inference___backward_cudnn_lstm_72668_72844
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis&
"gradients_transpose_grad_transpose%
!gradients_expanddims_grad_reshape'
#gradients_expanddims_1_grad_reshape
gradients_split_grad_concat!
gradients_split_1_grad_concat!
gradients_concat_grad_slice_1?^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2m
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :??????????????????2`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:?????????2O
gradients/grad_ys_4Identityplaceholder_4*
_output_shapes
: *
T0}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
_output_shapes
:*
T0?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
shrink_axis_mask*4
_output_shapes"
 :??????????????????2*
Index0*
T0?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :??????????????????2?
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:?????????2?
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*
shrink_axis_mask*+
_output_shapes
:?????????2*
T0*
Index0?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*4
_output_shapes"
 :??????????????????2c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
_output_shapes
:*
T0?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*i
_output_shapesW
U:??????????????????:?????????2:?????????2:?R*
T0?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :??????????????????u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
_output_shapes
:*
T0?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*'
_output_shapes
:?????????2*
T0y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*'
_output_shapes
:?????????2*
T0^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: ?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: g
gradients/concat_1_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
valueB:2*
dtype0i
gradients/concat_1_grad/Shape_2Const*
valueB:2*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_3Const*
valueB:2*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_4Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_5Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_6Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_7Const*
valueB:?*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_8Const*
valueB:2*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
valueB:2*
dtype0j
 gradients/concat_1_grad/Shape_11Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_12Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_13Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
valueB:2*
dtype0j
 gradients/concat_1_grad/Shape_15Const*
valueB:2*
dtype0*
_output_shapes
:?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:2?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
_output_shapes
:2*
Index0*
T0?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
_output_shapes
:2*
Index0*
T0?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes
:2?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
_output_shapes
:2*
Index0*
T0?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
_output_shapes
:2*
Index0*
T0?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
_output_shapes
:2*
Index0*
T0?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:2m
gradients/Reshape_grad/ShapeConst*
valueB"2      *
dtype0*
_output_shapes
:?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
_output_shapes

:2*
T0o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
valueB"2      *
dtype0?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_2_grad/ShapeConst*
valueB"2      *
dtype0*
_output_shapes
:?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
valueB"2      *
dtype0?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_4_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:22o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
valueB"2   2   *
dtype0?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
_output_shapes

:22*
T0o
gradients/Reshape_6_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:22o
gradients/Reshape_7_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:22h
gradients/Reshape_8_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
_output_shapes
:2*
T0h
gradients/Reshape_9_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
_output_shapes
:2*
T0i
gradients/Reshape_10_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_11_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_12_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:2?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_13_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
_output_shapes
:2*
T0i
gradients/Reshape_14_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
valueB:2*
dtype0?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:2?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
_output_shapes

:2*
T0?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
_output_shapes

:2*
T0?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
_output_shapes

:22*
T0?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:22?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
_output_shapes

:22*
T0?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:22?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:??
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	??
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	2?\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: ?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:?*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
valueB:?*
dtype0?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:?"G
gradients_concat_grad_slice_1&gradients/concat_grad/Slice_1:output:0"C
gradients_split_grad_concat$gradients/split_grad/concat:output:0"S
#gradients_expanddims_1_grad_reshape,gradients/ExpandDims_1_grad/Reshape:output:0"L
"gradients_transpose_grad_transpose&gradients/transpose_grad/transpose:y:0"G
gradients_split_1_grad_concat&gradients/split_1_grad/concat:output:0"O
!gradients_expanddims_grad_reshape*gradients/ExpandDims_grad/Reshape:output:0*?
_input_shapes?
?:?????????2:??????????????????2:?????????2:?????????2: :??????????????????2:::::?????????2::::?????????2:::::??????????????????:?????????2:?????????2:?R::?????????2:?????????2: ::::::::: : : : *=
api_implements+)lstm_a73b20f0-c393-4dea-a4aa-de0c2ac2517c*
api_preferred_deviceGPU*5
forward_function_name__forward_cudnn_lstm_72670*
	_noinline(: : : : : : : : : : :  :! :" :# :$ :% :& :  : : : : : : : : :	 :
 : : : : : : : : : : : 
?H
?
__inference_standard_lstm_71145

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?whilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*4
_output_shapes"
 :??????????????????*
T0B
ShapeShapetranspose:y:0*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
_output_shapes
: *
valueB :
?????????*
dtype0?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: ?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"????   *
dtype0*
_output_shapes
:?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*'
_output_shapes
:?????????*
Index0*
T0*
shrink_axis_mask?
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	2?n
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
addAddMatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:???????????
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:?????????2*
T0S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????2U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2R
add_1Addmul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:?????????2*
T0K
Tanh_1Tanh	add_1:z:0*'
_output_shapes
:?????????2*
T0Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2h
TensorArrayV2_1/element_shapeConst*
valueB :
?????????*
dtype0*
_output_shapes
: ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
?????????*
dtype0*
_output_shapes
: T
while/loop_counterConst*
_output_shapes
: *
value	B : *
dtype0?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
bodyR
while_body_71042*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *
T
2*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_71041K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:?????????2*
T0^
while/Identity_5Identitywhile:output:5*'
_output_shapes
:?????????2*
T0M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"????2   *
dtype0*
_output_shapes
:?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :??????????????????2h
strided_slice_2/stackConst*
valueB:
?????????*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*'
_output_shapes
:?????????2*
T0*
Index0*
shrink_axis_maske
transpose_1/permConst*
dtype0*
_output_shapes
:*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2[
runtimeConst"/device:CPU:0*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2?

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*4
_output_shapes"
 :??????????????????2*
T0?

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2?

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:?????????2*
T0?

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
	_noinline(*e
_input_shapesT
R:??????????????????:?????????2:?????????2:::*=
api_implements+)lstm_7622bbb0-93a1-4322-8e09-ef1893ca536d*
api_preferred_deviceCPU22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?
?
while_cond_72925
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
G
LessLesstimeless_strided_slice*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::: : : : : : : : :	 :
 :  
?
?
&__inference_lstm_1_layer_call_fn_76729

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*,
_gradient_op_typePartitionedCall-76723*J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_76253*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????02:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
?>
?
__inference_cudnn_lstm_74551

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
dtype0*
value	B :u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0;

zeros_likeConst*
valueBd*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
dtype0*
value	B :;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
?????????*
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0k
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
dtype0*
valueB: C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
Index0*
T0C
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
_input_shapes *=
api_implements+)lstm_583a098d-9984-4e53-9565-7f3e396fc7ec*
api_preferred_deviceGPU*
	_noinline(26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h
?>
?
__inference_cudnn_lstm_70301

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
dtype0*
value	B :9
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*
T01
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueB?*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
	num_split*
T0>
Const_3Const*
valueB:
?????????*
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
dtype0*
valueB"       N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
dtype0*
valueB"       N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
Nk
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
Index0I
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
dtype0*
valueB:E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
T0*
Index0C
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapes *=
api_implements+)lstm_0e2cd939-c429-480a-8893-82e6e02dfe56*
api_preferred_deviceGPU*
	_noinline(26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?+
?
while_body_71042
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"????   *
dtype0*
_output_shapes
:?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0time:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:??????????
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	??
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	2?*
dtype0n
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
addAddMatMul:product:0MatMul_1:product:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
T0T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:?????????2*
T0V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:?????????2*
T0S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????2U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2R
add_1Addmul:z:0	mul_1:z:0*'
_output_shapes
:?????????2*
T0V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:?????????2*
T0K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????2Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemtensorarrayv2_1time	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: E
add_2Addtimeadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: S
add_3Addwhile_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: ?
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:?????????2*
T0?

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:  : : : : : : : : :	 :
 
?
?
while_cond_74336
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
G
LessLesstimeless_strided_slice*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :?????????:?????????: : ::::	 :
 :  : : : : : : : : 
?+
?
while_body_73861
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"????2   *
dtype0*
_output_shapes
:?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0time:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:?????????2?
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:2d?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????db
addAddMatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0G
ConstConst*
_output_shapes
: *
value	B :*
dtype0Q
split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:?????????*
T0S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????U
mul_1MulSigmoid:y:0Tanh:y:0*'
_output_shapes
:?????????*
T0R
add_1Addmul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:??????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemtensorarrayv2_1time	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: E
add_2Addtimeadd_2/y:output:0*
_output_shapes
: *
T0I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: S
add_3Addwhile_loop_counteradd_3/y:output:0*
_output_shapes
: *
T0?
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0?

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:??????????

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0" 
strided_slicestrided_slice_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :	 :
 :  : : : : : : : 
?5
?
C__inference_model_14_layer_call_and_return_conditional_losses_76867

lstm_input

tech_input/
+tech_dense_0_statefulpartitionedcall_args_1/
+tech_dense_0_statefulpartitionedcall_args_2/
+tech_dense_3_statefulpartitionedcall_args_1/
+tech_dense_3_statefulpartitionedcall_args_2)
%lstm_0_statefulpartitionedcall_args_1)
%lstm_0_statefulpartitionedcall_args_2)
%lstm_0_statefulpartitionedcall_args_3)
%lstm_1_statefulpartitionedcall_args_1)
%lstm_1_statefulpartitionedcall_args_2)
%lstm_1_statefulpartitionedcall_args_30
,dense_pooling_statefulpartitionedcall_args_10
,dense_pooling_statefulpartitionedcall_args_22
.dense_pooling_1_statefulpartitionedcall_args_12
.dense_pooling_1_statefulpartitionedcall_args_2,
(dense_out_statefulpartitionedcall_args_1,
(dense_out_statefulpartitionedcall_args_2
identity??!dense_out/StatefulPartitionedCall?%dense_pooling/StatefulPartitionedCall?'dense_pooling_1/StatefulPartitionedCall?lstm_0/StatefulPartitionedCall?lstm_1/StatefulPartitionedCall?$tech_dense_0/StatefulPartitionedCall?$tech_dense_3/StatefulPartitionedCall?
$tech_dense_0/StatefulPartitionedCallStatefulPartitionedCall
tech_input+tech_dense_0_statefulpartitionedcall_args_1+tech_dense_0_statefulpartitionedcall_args_2*P
fKRI
G__inference_tech_dense_0_layer_call_and_return_conditional_losses_74752*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????2*,
_gradient_op_typePartitionedCall-74758?
tech_relu_0/PartitionedCallPartitionedCall-tech_dense_0/StatefulPartitionedCall:output:0*
Tin
2*'
_output_shapes
:?????????2*,
_gradient_op_typePartitionedCall-74780*O
fJRH
F__inference_tech_relu_0_layer_call_and_return_conditional_losses_74774*
Tout
2**
config_proto

GPU 

CPU2J 8?
lstm_input_1/PartitionedCallPartitionedCall
lstm_input*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:?????????0*,
_gradient_op_typePartitionedCall-74797*N
fIRG
E__inference_lstm_input_layer_call_and_return_conditional_losses_74791?
$tech_dense_3/StatefulPartitionedCallStatefulPartitionedCall$tech_relu_0/PartitionedCall:output:0+tech_dense_3_statefulpartitionedcall_args_1+tech_dense_3_statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-74820*P
fKRI
G__inference_tech_dense_3_layer_call_and_return_conditional_losses_74814?
lstm_0/StatefulPartitionedCallStatefulPartitionedCall%lstm_input_1/PartitionedCall:output:0%lstm_0_statefulpartitionedcall_args_1%lstm_0_statefulpartitionedcall_args_2%lstm_0_statefulpartitionedcall_args_3*
Tin
2*+
_output_shapes
:?????????02*,
_gradient_op_typePartitionedCall-75767*J
fERC
A__inference_lstm_0_layer_call_and_return_conditional_losses_75297*
Tout
2**
config_proto

GPU 

CPU2J 8?
lstm_1/StatefulPartitionedCallStatefulPartitionedCall'lstm_0/StatefulPartitionedCall:output:0%lstm_1_statefulpartitionedcall_args_1%lstm_1_statefulpartitionedcall_args_2%lstm_1_statefulpartitionedcall_args_3*
Tin
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-76723*J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_76253*
Tout
2**
config_proto

GPU 

CPU2J 8?
tech_relu_3/PartitionedCallPartitionedCall-tech_dense_3/StatefulPartitionedCall:output:0*'
_output_shapes
:?????????*
Tin
2*,
_gradient_op_typePartitionedCall-76754*O
fJRH
F__inference_tech_relu_3_layer_call_and_return_conditional_losses_76748*
Tout
2**
config_proto

GPU 

CPU2J 8?
concatenate/PartitionedCallPartitionedCall'lstm_1/StatefulPartitionedCall:output:0$tech_relu_3/PartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????2*
Tin
2*,
_gradient_op_typePartitionedCall-76775*O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_76768*
Tout
2?
%dense_pooling/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0,dense_pooling_statefulpartitionedcall_args_1,dense_pooling_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-76799*Q
fLRJ
H__inference_dense_pooling_layer_call_and_return_conditional_losses_76793*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????*
Tin
2?
'dense_pooling_1/StatefulPartitionedCallStatefulPartitionedCall.dense_pooling/StatefulPartitionedCall:output:0.dense_pooling_1_statefulpartitionedcall_args_1.dense_pooling_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-76827*S
fNRL
J__inference_dense_pooling_1_layer_call_and_return_conditional_losses_76821*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????
*
Tin
2?
!dense_out/StatefulPartitionedCallStatefulPartitionedCall0dense_pooling_1/StatefulPartitionedCall:output:0(dense_out_statefulpartitionedcall_args_1(dense_out_statefulpartitionedcall_args_2*'
_output_shapes
:?????????*
Tin
2*,
_gradient_op_typePartitionedCall-76855*M
fHRF
D__inference_dense_out_layer_call_and_return_conditional_losses_76849*
Tout
2**
config_proto

GPU 

CPU2J 8?
IdentityIdentity*dense_out/StatefulPartitionedCall:output:0"^dense_out/StatefulPartitionedCall&^dense_pooling/StatefulPartitionedCall(^dense_pooling_1/StatefulPartitionedCall^lstm_0/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall%^tech_dense_0/StatefulPartitionedCall%^tech_dense_3/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*}
_input_shapesl
j:?????????0:?????????::::::::::::::::2F
!dense_out/StatefulPartitionedCall!dense_out/StatefulPartitionedCall2N
%dense_pooling/StatefulPartitionedCall%dense_pooling/StatefulPartitionedCall2@
lstm_0/StatefulPartitionedCalllstm_0/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2L
$tech_dense_0/StatefulPartitionedCall$tech_dense_0/StatefulPartitionedCall2L
$tech_dense_3/StatefulPartitionedCall$tech_dense_3/StatefulPartitionedCall2R
'dense_pooling_1/StatefulPartitionedCall'dense_pooling_1/StatefulPartitionedCall:* &
$
_user_specified_name
lstm_input:*&
$
_user_specified_name
tech_input: : : : : : : :	 :
 : : : : : : : 
?
?
A__inference_lstm_1_layer_call_and_return_conditional_losses_73319

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
_output_shapes
: *
value	B :*
dtype0_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
_output_shapes
: *
value
B :?*
dtype0Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:?????????*
T0O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :?*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
dtype0*
_output_shapes
: *
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:??????????
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*o
_output_shapes]
[:?????????:??????????????????:?????????:?????????: *,
_gradient_op_typePartitionedCall-73030*(
f#R!
__inference_standard_lstm_73029*
Tout	
2**
config_proto

GPU 

CPU2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????2:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
?
?
,__inference_tech_dense_0_layer_call_fn_74763

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-74758*P
fKRI
G__inference_tech_dense_0_layer_call_and_return_conditional_losses_74752*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????2*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?G
?
__inference_standard_lstm_76430

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?whilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:0?????????2B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: ?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"????2   *
dtype0*
_output_shapes
:?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*'
_output_shapes
:?????????2*
Index0*
T0*
shrink_axis_mask?
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:2d{
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0?
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????db
addAddMatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:?????????:?????????:?????????:?????????T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:?????????*
T0V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????S
mulMulSigmoid_1:y:0init_c*'
_output_shapes
:?????????*
T0N
TanhTanhsplit:output:2*'
_output_shapes
:?????????*
T0U
mul_1MulSigmoid:y:0Tanh:y:0*'
_output_shapes
:?????????*
T0R
add_1Addmul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:?????????*
T0K
Tanh_1Tanh	add_1:z:0*'
_output_shapes
:?????????*
T0Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*'
_output_shapes
:?????????*
T0h
TensorArrayV2_1/element_shapeConst*
valueB :
?????????*
dtype0*
_output_shapes
: ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :0*
dtype0*
_output_shapes
: T
while/loop_counterConst*
dtype0*
_output_shapes
: *
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_76324*
bodyR
while_body_76325*L
_output_shapes:
8: : : : :?????????:?????????: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:?????????*
T0^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:?????????M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"????   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:0?????????h
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
?????????a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*'
_output_shapes
:?????????*
T0*
Index0*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*!
valueB"          *
dtype0?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????0[
runtimeConst"/device:CPU:0*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:??????????

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:?????????0?

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:??????????

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:??????????

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*
	_noinline(*\
_input_shapesK
I:?????????02:?????????:?????????:::*=
api_implements+)lstm_d99063c4-b3b5-4528-b76f-dadf8da915f1*
api_preferred_deviceCPU22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?5
?
C__inference_model_14_layer_call_and_return_conditional_losses_76937

inputs
inputs_1/
+tech_dense_0_statefulpartitionedcall_args_1/
+tech_dense_0_statefulpartitionedcall_args_2/
+tech_dense_3_statefulpartitionedcall_args_1/
+tech_dense_3_statefulpartitionedcall_args_2)
%lstm_0_statefulpartitionedcall_args_1)
%lstm_0_statefulpartitionedcall_args_2)
%lstm_0_statefulpartitionedcall_args_3)
%lstm_1_statefulpartitionedcall_args_1)
%lstm_1_statefulpartitionedcall_args_2)
%lstm_1_statefulpartitionedcall_args_30
,dense_pooling_statefulpartitionedcall_args_10
,dense_pooling_statefulpartitionedcall_args_22
.dense_pooling_1_statefulpartitionedcall_args_12
.dense_pooling_1_statefulpartitionedcall_args_2,
(dense_out_statefulpartitionedcall_args_1,
(dense_out_statefulpartitionedcall_args_2
identity??!dense_out/StatefulPartitionedCall?%dense_pooling/StatefulPartitionedCall?'dense_pooling_1/StatefulPartitionedCall?lstm_0/StatefulPartitionedCall?lstm_1/StatefulPartitionedCall?$tech_dense_0/StatefulPartitionedCall?$tech_dense_3/StatefulPartitionedCall?
$tech_dense_0/StatefulPartitionedCallStatefulPartitionedCallinputs_1+tech_dense_0_statefulpartitionedcall_args_1+tech_dense_0_statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????2*,
_gradient_op_typePartitionedCall-74758*P
fKRI
G__inference_tech_dense_0_layer_call_and_return_conditional_losses_74752?
tech_relu_0/PartitionedCallPartitionedCall-tech_dense_0/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????2*,
_gradient_op_typePartitionedCall-74780*O
fJRH
F__inference_tech_relu_0_layer_call_and_return_conditional_losses_74774*
Tout
2?
lstm_input/PartitionedCallPartitionedCallinputs**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:?????????0*,
_gradient_op_typePartitionedCall-74797*N
fIRG
E__inference_lstm_input_layer_call_and_return_conditional_losses_74791*
Tout
2?
$tech_dense_3/StatefulPartitionedCallStatefulPartitionedCall$tech_relu_0/PartitionedCall:output:0+tech_dense_3_statefulpartitionedcall_args_1+tech_dense_3_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-74820*P
fKRI
G__inference_tech_dense_3_layer_call_and_return_conditional_losses_74814*
Tout
2?
lstm_0/StatefulPartitionedCallStatefulPartitionedCall#lstm_input/PartitionedCall:output:0%lstm_0_statefulpartitionedcall_args_1%lstm_0_statefulpartitionedcall_args_2%lstm_0_statefulpartitionedcall_args_3**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:?????????02*,
_gradient_op_typePartitionedCall-75767*J
fERC
A__inference_lstm_0_layer_call_and_return_conditional_losses_75297*
Tout
2?
lstm_1/StatefulPartitionedCallStatefulPartitionedCall'lstm_0/StatefulPartitionedCall:output:0%lstm_1_statefulpartitionedcall_args_1%lstm_1_statefulpartitionedcall_args_2%lstm_1_statefulpartitionedcall_args_3**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????*
Tin
2*,
_gradient_op_typePartitionedCall-76723*J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_76253*
Tout
2?
tech_relu_3/PartitionedCallPartitionedCall-tech_dense_3/StatefulPartitionedCall:output:0*
Tin
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-76754*O
fJRH
F__inference_tech_relu_3_layer_call_and_return_conditional_losses_76748*
Tout
2**
config_proto

GPU 

CPU2J 8?
concatenate/PartitionedCallPartitionedCall'lstm_1/StatefulPartitionedCall:output:0$tech_relu_3/PartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????2*
Tin
2*,
_gradient_op_typePartitionedCall-76775*O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_76768*
Tout
2?
%dense_pooling/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0,dense_pooling_statefulpartitionedcall_args_1,dense_pooling_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????*
Tin
2*,
_gradient_op_typePartitionedCall-76799*Q
fLRJ
H__inference_dense_pooling_layer_call_and_return_conditional_losses_76793*
Tout
2?
'dense_pooling_1/StatefulPartitionedCallStatefulPartitionedCall.dense_pooling/StatefulPartitionedCall:output:0.dense_pooling_1_statefulpartitionedcall_args_1.dense_pooling_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-76827*S
fNRL
J__inference_dense_pooling_1_layer_call_and_return_conditional_losses_76821*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????
*
Tin
2?
!dense_out/StatefulPartitionedCallStatefulPartitionedCall0dense_pooling_1/StatefulPartitionedCall:output:0(dense_out_statefulpartitionedcall_args_1(dense_out_statefulpartitionedcall_args_2*'
_output_shapes
:?????????*
Tin
2*,
_gradient_op_typePartitionedCall-76855*M
fHRF
D__inference_dense_out_layer_call_and_return_conditional_losses_76849*
Tout
2**
config_proto

GPU 

CPU2J 8?
IdentityIdentity*dense_out/StatefulPartitionedCall:output:0"^dense_out/StatefulPartitionedCall&^dense_pooling/StatefulPartitionedCall(^dense_pooling_1/StatefulPartitionedCall^lstm_0/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall%^tech_dense_0/StatefulPartitionedCall%^tech_dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*}
_input_shapesl
j:?????????0:?????????::::::::::::::::2F
!dense_out/StatefulPartitionedCall!dense_out/StatefulPartitionedCall2N
%dense_pooling/StatefulPartitionedCall%dense_pooling/StatefulPartitionedCall2@
lstm_0/StatefulPartitionedCalllstm_0/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2L
$tech_dense_0/StatefulPartitionedCall$tech_dense_0/StatefulPartitionedCall2L
$tech_dense_3/StatefulPartitionedCall$tech_dense_3/StatefulPartitionedCall2R
'dense_pooling_1/StatefulPartitionedCall'dense_pooling_1/StatefulPartitionedCall: :& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs: : : : : : : :	 :
 : : : : : : 
?P
?
__forward_cudnn_lstm_75588

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
dtype0*
value	B : D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueB?*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
dtype0*
value	B : n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
?????????*
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
No
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0")
transpose_permtranspose/perm:output:0"!

expanddimsExpandDims:output:0"-
transpose_4_permtranspose_4/perm:output:0"%
expanddims_1ExpandDims_1:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0"
concat_1concat_1_0:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_9_permtranspose_9/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"-
transpose_3_permtranspose_3/perm:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"-
transpose_8_permtranspose_8/perm:output:0"
identityIdentity:output:0"3
strided_slice_stackstrided_slice/stack:output:0"-
transpose_2_permtranspose_2/perm:output:0"'
concat_1_axisconcat_1/axis:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"
init_hinit_h_0"7
strided_slice_stack_1strided_slice/stack_1:output:0"-
transpose_6_permtranspose_6/perm:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"-
transpose_5_permtranspose_5/perm:output:0*
	_noinline(*I
backward_function_name/-__inference___backward_cudnn_lstm_75586_75762*
_input_shapes *=
api_implements+)lstm_7f56906d-3660-42c1-afeb-29d9e0ca9efc*
api_preferred_deviceGPU26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?
?
while_cond_76324
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
less_1_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
[
LessLesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: K
Less_1Lesstimeless_1_strided_slice*
T0*
_output_shapes
: F

LogicalAnd
LogicalAndLess:z:0
Less_1:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::: : :	 :
 :  : : : : : : 
?P
?
__forward_cudnn_lstm_75121

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueB?*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
?????????*
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
dtype0*
value	B : ?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
No
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
Index0*
T0C
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_3Identity_3:output:0"-
transpose_8_permtranspose_8/perm:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"3
strided_slice_stackstrided_slice/stack:output:0"-
transpose_2_permtranspose_2/perm:output:0"'
concat_1_axisconcat_1/axis:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"
	transposetranspose_0:y:0"+
split_split_dimsplit/split_dim:output:0"#
concat_axisconcat/axis:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"-
transpose_5_permtranspose_5/perm:output:0")
transpose_permtranspose/perm:output:0"!

expanddimsExpandDims:output:0"-
transpose_4_permtranspose_4/perm:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"%
expanddims_1ExpandDims_1:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"
concat_1concat_1_0:output:0"-
transpose_9_permtranspose_9/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"-
transpose_3_permtranspose_3/perm:output:0*=
api_implements+)lstm_68d0f6f7-76c5-4200-8d0a-18146dc4d807*
api_preferred_deviceGPU*
	_noinline(*I
backward_function_name/-__inference___backward_cudnn_lstm_75119_75295*
_input_shapes 2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?
?
#__inference_signature_wrapper_77046

lstm_input

tech_input"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
lstm_input
tech_inputstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17*,
_gradient_op_typePartitionedCall-77027*)
f$R"
 __inference__wrapped_model_70967*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*}
_input_shapesl
j:?????????0:?????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:*&
$
_user_specified_name
tech_input: : : : : : : :	 :
 : : : : : : : :* &
$
_user_specified_name
lstm_input
?
?
while_cond_75368
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
less_1_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
[
LessLesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: K
Less_1Lesstimeless_1_strided_slice*
T0*
_output_shapes
: F

LogicalAnd
LogicalAndLess:z:0
Less_1:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::: : : :	 :
 :  : : : : : 
??
?
-__inference___backward_cudnn_lstm_70302_70478
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis&
"gradients_transpose_grad_transpose%
!gradients_expanddims_grad_reshape'
#gradients_expanddims_1_grad_reshape
gradients_split_grad_concat!
gradients_split_1_grad_concat!
gradients_concat_grad_slice_1?^
gradients/grad_ys_0Identityplaceholder*'
_output_shapes
:?????????2*
T0d
gradients/grad_ys_1Identityplaceholder_1*+
_output_shapes
:?????????02*
T0`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????2`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:?????????2O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*+
_output_shapes
:0?????????2*
Index0*
T0*
shrink_axis_mask?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:0?????????2?
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
T0*
Index0*
shrink_axis_mask*+
_output_shapes
:?????????2?
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:?????????2?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*+
_output_shapes
:0?????????2*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
Nc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
_output_shapes
:*
T0?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*
T0*`
_output_shapesN
L:0?????????:?????????2:?????????2:?R?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:?????????0u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
_output_shapes
:*
T0?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*'
_output_shapes
:?????????2*
T0y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
_output_shapes
:*
T0?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2^
gradients/concat_1_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: g
gradients/concat_1_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_1Const*
valueB:2*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_2Const*
dtype0*
_output_shapes
:*
valueB:2i
gradients/concat_1_grad/Shape_3Const*
dtype0*
_output_shapes
:*
valueB:2j
gradients/concat_1_grad/Shape_4Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_5Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_6Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_7Const*
valueB:?*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
valueB:2*
dtype0i
gradients/concat_1_grad/Shape_9Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_11Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_12Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_13Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_14Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_15Const*
valueB:2*
dtype0*
_output_shapes
:?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
_output_shapes
:2*
Index0*
T0?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
_output_shapes
:2*
Index0*
T0?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
_output_shapes
:2*
Index0*
T0?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes
:2?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:2?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
_output_shapes
:2*
Index0*
T0?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
_output_shapes
:2*
Index0*
T0?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
_output_shapes
:2*
Index0*
T0?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
_output_shapes
:2*
Index0*
T0?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:2m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
valueB"2      *
dtype0?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_1_grad/ShapeConst*
valueB"2      *
dtype0*
_output_shapes
:?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_2_grad/ShapeConst*
valueB"2      *
dtype0*
_output_shapes
:?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_3_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"2      ?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
_output_shapes

:2*
T0o
gradients/Reshape_4_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:22o
gradients/Reshape_5_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:22o
gradients/Reshape_6_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
_output_shapes

:22*
T0o
gradients/Reshape_7_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
_output_shapes

:22*
T0h
gradients/Reshape_8_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:2?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
_output_shapes
:2*
T0h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
valueB:2*
dtype0?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_10_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
valueB:2*
dtype0?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_12_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:2?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
_output_shapes
:2*
T0i
gradients/Reshape_13_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
_output_shapes
:2*
T0i
gradients/Reshape_14_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
_output_shapes
:2*
T0i
gradients/Reshape_15_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:2?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
_output_shapes

:22*
T0?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
_output_shapes

:22*
T0?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:22?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:22?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:??
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	??
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
_output_shapes
:	2?*
T0*
N\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: ?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:?*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:?*
dtype0*
_output_shapes
:?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
_output_shapes	
:?*
Index0*
T0"G
gradients_concat_grad_slice_1&gradients/concat_grad/Slice_1:output:0"C
gradients_split_grad_concat$gradients/split_grad/concat:output:0"S
#gradients_expanddims_1_grad_reshape,gradients/ExpandDims_1_grad/Reshape:output:0"L
"gradients_transpose_grad_transpose&gradients/transpose_grad/transpose:y:0"G
gradients_split_1_grad_concat&gradients/split_1_grad/concat:output:0"O
!gradients_expanddims_grad_reshape*gradients/ExpandDims_grad/Reshape:output:0*5
forward_function_name__forward_cudnn_lstm_70304*
	_noinline(*?
_input_shapes?
?:?????????2:?????????02:?????????2:?????????2: :0?????????2:::::?????????2::::?????????2:::::0?????????:?????????2:?????????2:?R::?????????2:?????????2: ::::::::: : : : *=
api_implements+)lstm_0e2cd939-c429-480a-8893-82e6e02dfe56*
api_preferred_deviceGPU: : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :  : : : : : : 
?+
?
while_body_70085
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"????   *
dtype0*
_output_shapes
:?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0time:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:??????????
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?*
dtype0?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	2?n
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
addAddMatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:???????????
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????G
ConstConst*
_output_shapes
: *
value	B :*
dtype0Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2N
TanhTanhsplit:output:2*'
_output_shapes
:?????????2*
T0U
mul_1MulSigmoid:y:0Tanh:y:0*'
_output_shapes
:?????????2*
T0R
add_1Addmul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????2Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????2?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemtensorarrayv2_1time	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: E
add_2Addtimeadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
_output_shapes
: *
value	B :*
dtype0S
add_3Addwhile_loop_counteradd_3/y:output:0*
_output_shapes
: *
T0?
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0?

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0?

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:?????????2*
T0?

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0" 
strided_slicestrided_slice_0"!

identity_1Identity_1:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:
 :  : : : : : : : : :	 
?
?
-__inference_dense_pooling_layer_call_fn_76804

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*Q
fLRJ
H__inference_dense_pooling_layer_call_and_return_conditional_losses_76793*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-76799?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
G
+__inference_tech_relu_0_layer_call_fn_74783

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*'
_output_shapes
:?????????2*,
_gradient_op_typePartitionedCall-74780*O
fJRH
F__inference_tech_relu_0_layer_call_and_return_conditional_losses_74774*
Tout
2**
config_proto

GPU 

CPU2J 8`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:?????????2*
T0"
identityIdentity:output:0*&
_input_shapes
:?????????2:& "
 
_user_specified_nameinputs
?
?
,__inference_tech_dense_3_layer_call_fn_74825

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-74820*P
fKRI
G__inference_tech_dense_3_layer_call_and_return_conditional_losses_74814*
Tout
2**
config_proto

GPU 

CPU2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs
??
?
-__inference___backward_cudnn_lstm_75119_75295
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis&
"gradients_transpose_grad_transpose%
!gradients_expanddims_grad_reshape'
#gradients_expanddims_1_grad_reshape
gradients_split_grad_concat!
gradients_split_1_grad_concat!
gradients_concat_grad_slice_1?^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:?????????02`
gradients/grad_ys_2Identityplaceholder_2*'
_output_shapes
:?????????2*
T0`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:?????????2O
gradients/grad_ys_4Identityplaceholder_4*
_output_shapes
: *
T0}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*+
_output_shapes
:0?????????2*
T0*
Index0*
shrink_axis_mask?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:0?????????2?
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*+
_output_shapes
:?????????2*
Index0*
T0*
shrink_axis_mask?
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*
shrink_axis_mask*+
_output_shapes
:?????????2*
T0*
Index0?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*+
_output_shapes
:0?????????2c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*
T0*`
_output_shapesN
L:0?????????:?????????2:?????????2:?R?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*+
_output_shapes
:?????????0*
T0u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
_output_shapes
:*
T0?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
_output_shapes
:*
T0?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: ?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: g
gradients/concat_1_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_1Const*
valueB:2*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_2Const*
valueB:2*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_3Const*
valueB:2*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_4Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_5Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_6Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_7Const*
dtype0*
_output_shapes
:*
valueB:?i
gradients/concat_1_grad/Shape_8Const*
valueB:2*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
valueB:2*
dtype0j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
valueB:2*
dtype0j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
valueB:2*
dtype0j
 gradients/concat_1_grad/Shape_12Const*
dtype0*
_output_shapes
:*
valueB:2j
 gradients/concat_1_grad/Shape_13Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
valueB:2*
dtype0j
 gradients/concat_1_grad/Shape_15Const*
dtype0*
_output_shapes
:*
valueB:2?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*t
_output_shapesb
`::::::::::::::::*
N?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
_output_shapes
:2*
Index0*
T0?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
_output_shapes
:2*
Index0*
T0?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes
:2?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes
:2?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:2?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
_output_shapes
:2*
Index0*
T0?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:2m
gradients/Reshape_grad/ShapeConst*
valueB"2      *
dtype0*
_output_shapes
:?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"2      ?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_2_grad/ShapeConst*
valueB"2      *
dtype0*
_output_shapes
:?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_3_grad/ShapeConst*
valueB"2      *
dtype0*
_output_shapes
:?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
_output_shapes

:2*
T0o
gradients/Reshape_4_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:22o
gradients/Reshape_5_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:22o
gradients/Reshape_6_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:22o
gradients/Reshape_7_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
_output_shapes

:22*
T0h
gradients/Reshape_8_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
_output_shapes
:2*
T0h
gradients/Reshape_9_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_10_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_11_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
_output_shapes
:2*
T0i
gradients/Reshape_12_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_13_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_14_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:2?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
_output_shapes
:2*
T0i
gradients/Reshape_15_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:2?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:2?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
_output_shapes

:2*
T0?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
_output_shapes

:22*
T0?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
_output_shapes

:22*
T0?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
_output_shapes

:22*
T0?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:22?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:??
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
_output_shapes
:	?*
T0?
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
_output_shapes
:	2?*
T0\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: ?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
_output_shapes
: *
T0f
gradients/concat_grad/ShapeConst*
valueB:?*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:?*
dtype0*
_output_shapes
:?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:?"G
gradients_split_1_grad_concat&gradients/split_1_grad/concat:output:0"O
!gradients_expanddims_grad_reshape*gradients/ExpandDims_grad/Reshape:output:0"G
gradients_concat_grad_slice_1&gradients/concat_grad/Slice_1:output:0"C
gradients_split_grad_concat$gradients/split_grad/concat:output:0"S
#gradients_expanddims_1_grad_reshape,gradients/ExpandDims_1_grad/Reshape:output:0"L
"gradients_transpose_grad_transpose&gradients/transpose_grad/transpose:y:0*?
_input_shapes?
?:?????????2:?????????02:?????????2:?????????2: :0?????????2:::::?????????2::::?????????2:::::0?????????:?????????2:?????????2:?R::?????????2:?????????2: ::::::::: : : : *=
api_implements+)lstm_68d0f6f7-76c5-4200-8d0a-18146dc4d807*
api_preferred_deviceGPU*5
forward_function_name__forward_cudnn_lstm_75121*
	_noinline(: :  :! :" :# :$ :% :& :  : : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : 
?P
?
__forward_cudnn_lstm_74554

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*
T0l
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0;

zeros_likeConst*
valueBd*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
	num_split*
T0>
Const_3Const*
valueB:
?????????*
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
dtype0*
valueB"       L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
No
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
Index0I
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
dtype0*
valueB:E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"'
concat_1_axisconcat_1/axis:output:0"
	transposetranspose_0:y:0"+
split_split_dimsplit/split_dim:output:0"#
concat_axisconcat/axis:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"
init_hinit_h_0"-
transpose_6_permtranspose_6/perm:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0")
transpose_permtranspose/perm:output:0"!

expanddimsExpandDims:output:0"-
transpose_4_permtranspose_4/perm:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"%
expanddims_1ExpandDims_1:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0"
concat_1concat_1_0:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_9_permtranspose_9/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"!

identity_1Identity_1:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"-
transpose_8_permtranspose_8/perm:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"-
transpose_2_permtranspose_2/perm:output:0"3
strided_slice_stackstrided_slice/stack:output:0*I
backward_function_name/-__inference___backward_cudnn_lstm_74552_74728*
_input_shapes *=
api_implements+)lstm_583a098d-9984-4e53-9565-7f3e396fc7ec*
api_preferred_deviceGPU*
	_noinline(26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
??
?
-__inference___backward_cudnn_lstm_74076_74252
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis&
"gradients_transpose_grad_transpose%
!gradients_expanddims_grad_reshape'
#gradients_expanddims_1_grad_reshape
gradients_split_grad_concat!
gradients_split_1_grad_concat!
gradients_concat_grad_slice_1?^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????m
gradients/grad_ys_1Identityplaceholder_1*4
_output_shapes"
 :??????????????????*
T0`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????`
gradients/grad_ys_3Identityplaceholder_3*'
_output_shapes
:?????????*
T0O
gradients/grad_ys_4Identityplaceholder_4*
_output_shapes
: *
T0}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*4
_output_shapes"
 :??????????????????*
Index0*
T0*
shrink_axis_mask?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*4
_output_shapes"
 :??????????????????*
T0?
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*+
_output_shapes
:?????????*
T0*
Index0*
shrink_axis_mask?
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:??????????
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*4
_output_shapes"
 :??????????????????*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
Nc
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
_output_shapes
:*
T0?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*i
_output_shapesW
U:??????????????????2:?????????:?????????:?<*
T0?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*4
_output_shapes"
 :??????????????????2*
T0u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*'
_output_shapes
:?????????*
T0^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: ?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
valueB:?	*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_1Const*
valueB:?	*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_2Const*
valueB:?	*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_3Const*
dtype0*
_output_shapes
:*
valueB:?	j
gradients/concat_1_grad/Shape_4Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_5Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_6Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_7Const*
valueB:?*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_8Const*
valueB:*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_11Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_12Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
valueB:*
dtype0j
 gradients/concat_1_grad/Shape_14Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
valueB:*
dtype0?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
_output_shapes	
:?	*
Index0*
T0?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:?	?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:?	?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:?	?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
_output_shapes
:*
Index0*
T0?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
_output_shapes
:*
Index0*
T0?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:m
gradients/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"   2   ?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_1_grad/ShapeConst*
valueB"   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_2_grad/ShapeConst*
valueB"   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
_output_shapes

:2*
T0o
gradients/Reshape_3_grad/ShapeConst*
valueB"   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_4_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_5_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
_output_shapes

:*
T0o
gradients/Reshape_6_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_7_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:h
gradients/Reshape_8_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
_output_shapes
:*
T0i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_11_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_12_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_13_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_14_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_15_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
_output_shapes
:*
T0?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
_output_shapes	
:?*
T0*
N?
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
_output_shapes

:2d*
T0*
N?
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes

:d\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: ?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: e
gradients/concat_grad/ShapeConst*
_output_shapes
:*
valueB:d*
dtype0g
gradients/concat_grad/Shape_1Const*
valueB:d*
dtype0*
_output_shapes
:?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
_output_shapes
:d*
Index0*
T0?
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:d"S
#gradients_expanddims_1_grad_reshape,gradients/ExpandDims_1_grad/Reshape:output:0"L
"gradients_transpose_grad_transpose&gradients/transpose_grad/transpose:y:0"G
gradients_split_1_grad_concat&gradients/split_1_grad/concat:output:0"O
!gradients_expanddims_grad_reshape*gradients/ExpandDims_grad/Reshape:output:0"G
gradients_concat_grad_slice_1&gradients/concat_grad/Slice_1:output:0"C
gradients_split_grad_concat$gradients/split_grad/concat:output:0*=
api_implements+)lstm_f5fde02f-8c16-4750-9be2-71a2dc99d757*
api_preferred_deviceGPU*5
forward_function_name__forward_cudnn_lstm_74078*
	_noinline(*?
_input_shapes?
?:?????????:??????????????????:?????????:?????????: :??????????????????:::::?????????::::?????????:::::??????????????????2:?????????:?????????:?<::?????????:?????????: ::::::::: : : : :  : : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& 
??
?
-__inference___backward_cudnn_lstm_76075_76251
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis&
"gradients_transpose_grad_transpose%
!gradients_expanddims_grad_reshape'
#gradients_expanddims_1_grad_reshape
gradients_split_grad_concat!
gradients_split_1_grad_concat!
gradients_concat_grad_slice_1?^
gradients/grad_ys_0Identityplaceholder*'
_output_shapes
:?????????*
T0d
gradients/grad_ys_1Identityplaceholder_1*+
_output_shapes
:?????????0*
T0`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????`
gradients/grad_ys_3Identityplaceholder_3*'
_output_shapes
:?????????*
T0O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
shrink_axis_mask*+
_output_shapes
:0?????????*
T0*
Index0?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:0??????????
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
_output_shapes
:*
T0?
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*+
_output_shapes
:?????????*
Index0*
T0*
shrink_axis_mask?
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*+
_output_shapes
:?????????*
Index0*
T0*
shrink_axis_mask?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*+
_output_shapes
:0?????????c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*
T0*`
_output_shapesN
L:0?????????2:?????????:?????????:?<?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:?????????02u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
_output_shapes
:*
T0?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*'
_output_shapes
:?????????*
T0y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
_output_shapes
:*
T0?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:?????????^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: ?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
valueB:?	*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:?	j
gradients/concat_1_grad/Shape_2Const*
dtype0*
_output_shapes
:*
valueB:?	j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
valueB:?	*
dtype0j
gradients/concat_1_grad/Shape_4Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_5Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_6Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_7Const*
valueB:?*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_8Const*
valueB:*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_11Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_12Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_13Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_14Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_15Const*
valueB:*
dtype0*
_output_shapes
:?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
_output_shapes	
:?	*
Index0*
T0?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
_output_shapes	
:?	*
Index0*
T0?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:?	?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
_output_shapes	
:?	*
Index0*
T0?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
_output_shapes
:*
Index0*
T0?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
_output_shapes
:*
Index0*
T0?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
_output_shapes
:*
Index0*
T0m
gradients/Reshape_grad/ShapeConst*
valueB"   2   *
dtype0*
_output_shapes
:?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_1_grad/ShapeConst*
valueB"   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_2_grad/ShapeConst*
valueB"   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_3_grad/ShapeConst*
valueB"   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_4_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"      ?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_5_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_6_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_7_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:h
gradients/Reshape_8_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
_output_shapes
:*
T0h
gradients/Reshape_9_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_10_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_11_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_12_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
_output_shapes
:*
T0i
gradients/Reshape_13_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_14_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_15_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
_output_shapes

:2*
T0?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
_output_shapes

:2*
T0?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
_output_shapes

:2*
T0?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
_output_shapes	
:?*
T0?
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes

:2d?
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes

:d\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: ?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: e
gradients/concat_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_1Const*
valueB:d*
dtype0*
_output_shapes
:?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:d?
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:d"G
gradients_split_1_grad_concat&gradients/split_1_grad/concat:output:0"O
!gradients_expanddims_grad_reshape*gradients/ExpandDims_grad/Reshape:output:0"C
gradients_split_grad_concat$gradients/split_grad/concat:output:0"G
gradients_concat_grad_slice_1&gradients/concat_grad/Slice_1:output:0"S
#gradients_expanddims_1_grad_reshape,gradients/ExpandDims_1_grad/Reshape:output:0"L
"gradients_transpose_grad_transpose&gradients/transpose_grad/transpose:y:0*?
_input_shapes?
?:?????????:?????????0:?????????:?????????: :0?????????:::::?????????::::?????????:::::0?????????2:?????????:?????????:?<::?????????:?????????: ::::::::: : : : *=
api_implements+)lstm_07d2f4a3-a6ab-4ba4-8910-49922c38a5c9*
api_preferred_deviceGPU*5
forward_function_name__forward_cudnn_lstm_76077*
	_noinline(: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :  
?
?
A__inference_lstm_0_layer_call_and_return_conditional_losses_75764

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :2*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :?*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :2*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2O
zeros_1/mul/yConst*
_output_shapes
: *
value	B :2*
dtype0c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :?*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
value	B :2*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*'
_output_shapes
:?????????2*
T0?
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*,
_gradient_op_typePartitionedCall-75475*(
f#R!
__inference_standard_lstm_75474*
Tout	
2**
config_proto

GPU 

CPU2J 8*f
_output_shapesT
R:?????????2:?????????02:?????????2:?????????2: *
Tin

2?
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????02"
identityIdentity:output:0*6
_input_shapes%
#:?????????0:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
?>
?
__inference_cudnn_lstm_76541

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*
T0l
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0;

zeros_likeConst*
valueBd*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
dtype0*
value	B : W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
	num_split*
T0>
Const_3Const*
valueB:
?????????*
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
dtype0*
valueB"       N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0k
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
Index0I
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
Index0*
T0C
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*
T0*
Index0C
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapes *=
api_implements+)lstm_d99063c4-b3b5-4528-b76f-dadf8da915f1*
api_preferred_deviceGPU*
	_noinline(20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?P
?
__forward_cudnn_lstm_70767

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axis??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*
T0l
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0;

zeros_likeConst*
valueBd*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
	num_split*
T0>
Const_3Const*
dtype0*
valueB:
?????????E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
dtype0*
valueB"       L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
No
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
dtype0*
valueB:
?????????C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
Index0I
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
dtype0*
valueB: E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*
Index0*
T0C
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"/
split_1_split_dimsplit_1/split_dim:output:0"-
transpose_3_permtranspose_3/perm:output:0"!

identity_1Identity_1:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"-
transpose_8_permtranspose_8/perm:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"3
strided_slice_stackstrided_slice/stack:output:0"-
transpose_2_permtranspose_2/perm:output:0"'
concat_1_axisconcat_1/axis:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"
	transposetranspose_0:y:0"+
split_split_dimsplit/split_dim:output:0"#
concat_axisconcat/axis:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0")
transpose_permtranspose/perm:output:0"!

expanddimsExpandDims:output:0"-
transpose_4_permtranspose_4/perm:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"%
expanddims_1ExpandDims_1:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0"
concat_1concat_1_0:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_9_permtranspose_9/perm:output:0*I
backward_function_name/-__inference___backward_cudnn_lstm_70765_70941*
_input_shapes *=
api_implements+)lstm_ffcd560d-1075-4e30-88b0-2ea61de6018e*
api_preferred_deviceGPU*
	_noinline(2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?H
?
__inference_standard_lstm_72080

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?whilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????B
ShapeShapetranspose:y:0*
_output_shapes
:*
T0]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0f
TensorArrayV2/element_shapeConst*
valueB :
?????????*
dtype0*
_output_shapes
: ?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
element_dtype0*
_output_shapes
: *

shape_type0?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
valueB"????   *
dtype0?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:?????????*
Index0*
T0?
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	2?n
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0c
addAddMatMul:product:0MatMul_1:product:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:?????????2*
T0V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????2U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????2R
add_1Addmul:z:0	mul_1:z:0*'
_output_shapes
:?????????2*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2K
Tanh_1Tanh	add_1:z:0*'
_output_shapes
:?????????2*
T0Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*'
_output_shapes
:?????????2*
T0h
TensorArrayV2_1/element_shapeConst*
valueB :
?????????*
dtype0*
_output_shapes
: ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
?????????*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
bodyR
while_body_71977*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *
T
2*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_71976K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:?????????2^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:?????????2M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"????2   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :??????????????????2h
strided_slice_2/stackConst*
valueB:
?????????*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:?????????2*
T0*
Index0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*4
_output_shapes"
 :??????????????????2*
T0[
runtimeConst"/device:CPU:0*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2?

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :??????????????????2?

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:?????????2*
T0?

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2?

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*
	_noinline(*e
_input_shapesT
R:??????????????????:?????????2:?????????2:::*=
api_implements+)lstm_6b86d02e-a5e5-4d08-8f29-1ea05985005f*
api_preferred_deviceCPU22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel
?
?
while_cond_75857
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
less_1_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
[
LessLesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: K
Less_1Lesstimeless_1_strided_slice*
_output_shapes
: *
T0F

LogicalAnd
LogicalAndLess:z:0
Less_1:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :?????????:?????????: : ::::  : : : : : : : : :	 :
 
?
?
while_cond_70084
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
less_1_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
[
LessLesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: K
Less_1Lesstimeless_1_strided_slice*
_output_shapes
: *
T0F

LogicalAnd
LogicalAndLess:z:0
Less_1:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :?????????2:?????????2: : ::::  : : : : : : : : :	 :
 
??
?
-__inference___backward_cudnn_lstm_71257_71433
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis&
"gradients_transpose_grad_transpose%
!gradients_expanddims_grad_reshape'
#gradients_expanddims_1_grad_reshape
gradients_split_grad_concat!
gradients_split_1_grad_concat!
gradients_concat_grad_slice_1?^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????2m
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :??????????????????2`
gradients/grad_ys_2Identityplaceholder_2*'
_output_shapes
:?????????2*
T0`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:?????????2O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
shrink_axis_mask*4
_output_shapes"
 :??????????????????2*
T0*
Index0?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :??????????????????2?
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
T0*
Index0*
shrink_axis_mask*+
_output_shapes
:?????????2?
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*+
_output_shapes
:?????????2*
T0*
Index0*
shrink_axis_mask?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*4
_output_shapes"
 :??????????????????2c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
_output_shapes
:*
T0?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*i
_output_shapesW
U:??????????????????:?????????2:?????????2:?R*
T0?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :??????????????????u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:?????????2y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*'
_output_shapes
:?????????2*
T0^
gradients/concat_1_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
_output_shapes
: *
T0g
gradients/concat_1_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_1Const*
valueB:2*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_2Const*
dtype0*
_output_shapes
:*
valueB:2i
gradients/concat_1_grad/Shape_3Const*
valueB:2*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_4Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_5Const*
dtype0*
_output_shapes
:*
valueB:?j
gradients/concat_1_grad/Shape_6Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_7Const*
valueB:?*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
valueB:2*
dtype0i
gradients/concat_1_grad/Shape_9Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_11Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_12Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_13Const*
valueB:2*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_14Const*
dtype0*
_output_shapes
:*
valueB:2j
 gradients/concat_1_grad/Shape_15Const*
valueB:2*
dtype0*
_output_shapes
:?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
_output_shapes
:2*
Index0*
T0?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes
:2?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
_output_shapes
:2*
Index0*
T0?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes
:2?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
_output_shapes
:2*
Index0*
T0?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
_output_shapes
:2*
Index0*
T0?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:2?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
_output_shapes
:2*
Index0*
T0?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
_output_shapes
:2*
Index0*
T0?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
_output_shapes
:2*
Index0*
T0?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
_output_shapes
:2*
Index0*
T0?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:2m
gradients/Reshape_grad/ShapeConst*
valueB"2      *
dtype0*
_output_shapes
:?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
_output_shapes

:2*
T0o
gradients/Reshape_1_grad/ShapeConst*
valueB"2      *
dtype0*
_output_shapes
:?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
_output_shapes

:2*
T0o
gradients/Reshape_2_grad/ShapeConst*
valueB"2      *
dtype0*
_output_shapes
:?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
_output_shapes

:2*
T0o
gradients/Reshape_3_grad/ShapeConst*
valueB"2      *
dtype0*
_output_shapes
:?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
_output_shapes

:2*
T0o
gradients/Reshape_4_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
_output_shapes

:22*
T0o
gradients/Reshape_5_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:22o
gradients/Reshape_6_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
_output_shapes

:22*
T0o
gradients/Reshape_7_grad/ShapeConst*
valueB"2   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:22h
gradients/Reshape_8_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:2h
gradients/Reshape_9_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
_output_shapes
:2*
T0i
gradients/Reshape_10_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_11_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
_output_shapes
:2*
T0i
gradients/Reshape_12_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_13_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:2i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
valueB:2*
dtype0?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
_output_shapes
:2*
T0i
gradients/Reshape_15_grad/ShapeConst*
valueB:2*
dtype0*
_output_shapes
:?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:2?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
_output_shapes

:2*
T0?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
_output_shapes

:2*
T0?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
_output_shapes

:2*
T0?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:22?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:22?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:22?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:22?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:??
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
_output_shapes
:	?*
T0?
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
_output_shapes
:	2?*
T0\
gradients/concat_grad/RankConst*
_output_shapes
: *
value	B :*
dtype0?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
_output_shapes
: *
T0f
gradients/concat_grad/ShapeConst*
valueB:?*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:??
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:?"G
gradients_concat_grad_slice_1&gradients/concat_grad/Slice_1:output:0"C
gradients_split_grad_concat$gradients/split_grad/concat:output:0"S
#gradients_expanddims_1_grad_reshape,gradients/ExpandDims_1_grad/Reshape:output:0"L
"gradients_transpose_grad_transpose&gradients/transpose_grad/transpose:y:0"G
gradients_split_1_grad_concat&gradients/split_1_grad/concat:output:0"O
!gradients_expanddims_grad_reshape*gradients/ExpandDims_grad/Reshape:output:0*?
_input_shapes?
?:?????????2:??????????????????2:?????????2:?????????2: :??????????????????2:::::?????????2::::?????????2:::::??????????????????:?????????2:?????????2:?R::?????????2:?????????2: ::::::::: : : : *=
api_implements+)lstm_7622bbb0-93a1-4322-8e09-ef1893ca536d*
api_preferred_deviceGPU*5
forward_function_name__forward_cudnn_lstm_71259*
	_noinline(: : : :  :! :" :# :$ :% :& :  : : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : 
?
?
while_cond_70547
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
less_1_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
[
LessLesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: K
Less_1Lesstimeless_1_strided_slice*
T0*
_output_shapes
: F

LogicalAnd
LogicalAndLess:z:0
Less_1:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :?????????:?????????: : ::::
 :  : : : : : : : : :	 
?
?
/__inference_dense_pooling_1_layer_call_fn_76832

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-76827*S
fNRL
J__inference_dense_pooling_1_layer_call_and_return_conditional_losses_76821*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????
*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
??
?
-__inference___backward_cudnn_lstm_73141_73317
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis&
"gradients_transpose_grad_transpose%
!gradients_expanddims_grad_reshape'
#gradients_expanddims_1_grad_reshape
gradients_split_grad_concat!
gradients_split_1_grad_concat!
gradients_concat_grad_slice_1?^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:?????????m
gradients/grad_ys_1Identityplaceholder_1*4
_output_shapes"
 :??????????????????*
T0`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:?????????`
gradients/grad_ys_3Identityplaceholder_3*'
_output_shapes
:?????????*
T0O
gradients/grad_ys_4Identityplaceholder_4*
_output_shapes
: *
T0}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
shrink_axis_mask*4
_output_shapes"
 :??????????????????*
T0*
Index0?
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:?
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*4
_output_shapes"
 :??????????????????*
T0?
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
_output_shapes
:*
T0?
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
shrink_axis_mask*+
_output_shapes
:?????????*
T0*
Index0?
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:?
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*
shrink_axis_mask*+
_output_shapes
:?????????*
T0*
Index0?
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*4
_output_shapes"
 :??????????????????c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:?
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*i
_output_shapesW
U:??????????????????2:?????????:?????????:?<*
T0?
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:?
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :??????????????????2u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:?
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*'
_output_shapes
:?????????*
T0y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:?
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:?????????^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: ?
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
_output_shapes
: *
T0h
gradients/concat_1_grad/ShapeConst*
valueB:?	*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_1Const*
valueB:?	*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
valueB:?	*
dtype0j
gradients/concat_1_grad/Shape_3Const*
valueB:?	*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_4Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_5Const*
valueB:?*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
valueB:?*
dtype0j
gradients/concat_1_grad/Shape_7Const*
valueB:?*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_8Const*
valueB:*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_11Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_12Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_13Const*
valueB:*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_14Const*
dtype0*
_output_shapes
:*
valueB:j
 gradients/concat_1_grad/Shape_15Const*
dtype0*
_output_shapes
:*
valueB:?
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::?
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:?	?
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:?	?
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
_output_shapes	
:?	*
Index0*
T0?
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
_output_shapes	
:?	*
Index0*
T0?
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
_output_shapes	
:?*
Index0*
T0?
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:??
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
_output_shapes
:*
Index0*
T0?
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
_output_shapes
:*
Index0*
T0?
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:?
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
_output_shapes
:*
Index0*
T0m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
valueB"   2   *
dtype0?
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
_output_shapes

:2*
T0o
gradients/Reshape_1_grad/ShapeConst*
valueB"   2   *
dtype0*
_output_shapes
:?
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:2o
gradients/Reshape_2_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"   2   ?
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
_output_shapes

:2*
T0o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
valueB"   2   *
dtype0?
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
_output_shapes

:2*
T0o
gradients/Reshape_4_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:?
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:o
gradients/Reshape_5_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"      ?
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
_output_shapes

:*
T0o
gradients/Reshape_6_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"      ?
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
_output_shapes

:*
T0o
gradients/Reshape_7_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:?
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:h
gradients/Reshape_8_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
_output_shapes
:*
T0h
gradients/Reshape_9_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
_output_shapes
:*
T0i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
valueB:*
dtype0?
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
_output_shapes
:*
T0i
gradients/Reshape_11_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
_output_shapes
:*
T0i
gradients/Reshape_12_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_13_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_14_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:?
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:i
gradients/Reshape_15_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:?
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
_output_shapes
:*
T0?
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:?
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
_output_shapes

:2*
T0?
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:?
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:?
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:?
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:2?
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:?
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:?
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:?
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
_output_shapes

:*
T0?
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:?
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
_output_shapes

:*
T0?
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:?
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
_output_shapes

:*
T0?
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:??
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes

:2d?
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
_output_shapes

:d*
T0\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: ?
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: e
gradients/concat_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:g
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
valueB:d*
dtype0?
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0* 
_output_shapes
::*
N?
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes
:d?
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
_output_shapes
:d*
Index0*
T0"G
gradients_split_1_grad_concat&gradients/split_1_grad/concat:output:0"O
!gradients_expanddims_grad_reshape*gradients/ExpandDims_grad/Reshape:output:0"G
gradients_concat_grad_slice_1&gradients/concat_grad/Slice_1:output:0"C
gradients_split_grad_concat$gradients/split_grad/concat:output:0"S
#gradients_expanddims_1_grad_reshape,gradients/ExpandDims_1_grad/Reshape:output:0"L
"gradients_transpose_grad_transpose&gradients/transpose_grad/transpose:y:0*?
_input_shapes?
?:?????????:??????????????????:?????????:?????????: :??????????????????:::::?????????::::?????????:::::??????????????????2:?????????:?????????:?<::?????????:?????????: ::::::::: : : : *=
api_implements+)lstm_3f27c04b-652c-4bbb-81f8-497acfa4408c*
api_preferred_deviceGPU*5
forward_function_name__forward_cudnn_lstm_73143*
	_noinline(:" :# :$ :% :& :  : : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! 
?
?
G__inference_tech_dense_3_layer_call_and_return_conditional_losses_74814

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:2i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????2::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?>
?
__inference_cudnn_lstm_70764

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
dtype0*
value	B :u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0;

zeros_likeConst*
valueBd*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T01
Const_2Const*
dtype0*
value	B :;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
dtype0*
valueB:
?????????E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
dtype0*
valueB"       N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
Nk
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
Index0I
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
T0*
Index0C
strided_slice_2/stackConst*
dtype0*
valueB: E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*
_input_shapes *=
api_implements+)lstm_ffcd560d-1075-4e30-88b0-2ea61de6018e*
api_preferred_deviceGPU*
	_noinline(26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?+
?
while_body_76325
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"????2   *
dtype0*
_output_shapes
:?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0time:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:?????????2?
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:2d?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????db
addAddMatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:dm
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:?????????*
T0S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????R
add_1Addmul:z:0	mul_1:z:0*'
_output_shapes
:?????????*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*'
_output_shapes
:?????????*
T0?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemtensorarrayv2_1time	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
dtype0*
_output_shapes
: *
value	B :E
add_2Addtimeadd_2/y:output:0*
_output_shapes
: *
T0I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: S
add_3Addwhile_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: ?
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:??????????

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0" 
strided_slicestrided_slice_0"!

identity_1Identity_1:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:  : : : : : : : : :	 :
 
?
?
A__inference_lstm_1_layer_call_and_return_conditional_losses_76720

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0M
zeros/mul/yConst*
value	B :*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :?*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????O
zeros_1/mul/yConst*
value	B :*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :?*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
value	B :*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:??????????
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5**
config_proto

GPU 

CPU2J 8*f
_output_shapesT
R:?????????:?????????0:?????????:?????????: *
Tin

2*,
_gradient_op_typePartitionedCall-76431*(
f#R!
__inference_standard_lstm_76430*
Tout	
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*6
_input_shapes%
#:?????????02:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
?G
?
__inference_standard_lstm_75474

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?whilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*+
_output_shapes
:0?????????*
T0B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
valueB :
?????????*
dtype0*
_output_shapes
: ?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: ?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"????   *
dtype0*
_output_shapes
:?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:??????????
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	2?n
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0c
addAddMatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:???????????
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:?????????2:?????????2:?????????2:?????????2T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????2N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????2U
mul_1MulSigmoid:y:0Tanh:y:0*'
_output_shapes
:?????????2*
T0R
add_1Addmul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2K
Tanh_1Tanh	add_1:z:0*'
_output_shapes
:?????????2*
T0Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*'
_output_shapes
:?????????2*
T0h
TensorArrayV2_1/element_shapeConst*
valueB :
?????????*
dtype0*
_output_shapes
: ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
dtype0*
_output_shapes
: *
value	B : Z
while/maximum_iterationsConst*
value	B :0*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
bodyR
while_body_75369*L
_output_shapes:
8: : : : :?????????2:?????????2: : : : : *
T
2*K
output_shapes:
8: : : : :?????????2:?????????2: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_75368K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:?????????2*
T0^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:?????????2M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"????2   *
dtype0*
_output_shapes
:?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:0?????????2h
strided_slice_2/stackConst*
valueB:
?????????*
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*'
_output_shapes
:?????????2*
Index0*
T0*
shrink_axis_maske
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*+
_output_shapes
:?????????02*
T0[
runtimeConst"/device:CPU:0*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:?????????2*
T0?

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:?????????02?

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2?

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2?

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*
	_noinline(*\
_input_shapesK
I:?????????0:?????????2:?????????2:::*=
api_implements+)lstm_7f56906d-3660-42c1-afeb-29d9e0ca9efc*
api_preferred_deviceCPU2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h
?
?
(__inference_model_14_layer_call_fn_76957

lstm_input

tech_input"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
lstm_input
tech_inputstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17*,
_gradient_op_typePartitionedCall-76938*L
fGRE
C__inference_model_14_layer_call_and_return_conditional_losses_76937*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*}
_input_shapesl
j:?????????0:?????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : :* &
$
_user_specified_name
lstm_input:*&
$
_user_specified_name
tech_input: : : : : : : :	 :
 
?>
?
__inference_cudnn_lstm_74075

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*
T0l
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0;

zeros_likeConst*
valueBd*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
?????????*
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
dtype0*
valueB"       L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
Nk
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
Index0I
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
Index0*
T0C
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*
T0*
Index0C
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*
_input_shapes *=
api_implements+)lstm_f5fde02f-8c16-4750-9be2-71a2dc99d757*
api_preferred_deviceGPU*
	_noinline(2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?>
?
__inference_cudnn_lstm_73140

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4??CudnnRNN?concat/ReadVariableOp?split/ReadVariableOp?split_1/ReadVariableOp?zeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*
T01
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*
T0l
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0;

zeros_likeConst*
valueBd*    *
dtype0?
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
	num_split*
T0>
Const_3Const*
valueB:
?????????*
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
dtype0*
valueB"       N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0?
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
Nk
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
?????????*
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0?
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskI
transpose_9/permConst*
dtype0*!
valueB"          O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0?
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0?
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0?
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0?

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*
_input_shapes *=
api_implements+)lstm_3f27c04b-652c-4bbb-81f8-497acfa4408c*
api_preferred_deviceGPU*
	_noinline(20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
?
?
A__inference_lstm_0_layer_call_and_return_conditional_losses_72370

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity??StatefulPartitionedCall;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskM
zeros/mul/yConst*
value	B :2*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
_output_shapes
: *
value
B :?*
dtype0Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :2*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
_output_shapes
:*
T0P
zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2O
zeros_1/mul/yConst*
value	B :2*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :?*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
value	B :2*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*'
_output_shapes
:?????????2*
T0?
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*,
_gradient_op_typePartitionedCall-72081*(
f#R!
__inference_standard_lstm_72080*
Tout	
2**
config_proto

GPU 

CPU2J 8*
Tin

2*o
_output_shapes]
[:?????????2:??????????????????2:?????????2:?????????2: ?
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
?
?
)__inference_dense_out_layer_call_fn_76860

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????*,
_gradient_op_typePartitionedCall-76855*M
fHRF
D__inference_dense_out_layer_call_and_return_conditional_losses_76849*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????
::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs
?+
?
while_body_72926
while_loop_counter
while_maximum_iterations
time
tensorarrayv2_1

init_h

init_c
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"????2   *
dtype0*
_output_shapes
:?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0time:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:?????????2?
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:2d*
dtype0?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:dm
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*'
_output_shapes
:?????????d*
T0b
addAddMatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:d*
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:?????????*
T0S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:?????????N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:?????????U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:?????????R
add_1Addmul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:?????????Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:??????????
$TensorArrayV2Write/TensorListSetItemTensorListSetItemtensorarrayv2_1time	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: E
add_2Addtimeadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: S
add_3Addwhile_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: ?
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ?

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:??????????

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0" 
strided_slicestrided_slice_0"!

identity_1Identity_1:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:  : : : : : : : : :	 :
 
?	
?
D__inference_dense_out_layer_call_and_return_conditional_losses_76849

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : "7L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*?
serving_default?
E

lstm_input7
serving_default_lstm_input:0?????????0
A

tech_input3
serving_default_tech_input:0?????????=
	dense_out0
StatefulPartitionedCall:0?????????tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:??
?^
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"?Y
_tf_keras_model?Y{"class_name": "Model", "name": "model_14", "trainable": true, "expects_training_arg": true, "dtype": null, "batch_input_shape": null, "config": {"name": "model_14", "layers": [{"name": "tech_input", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 7], "dtype": "float32", "sparse": false, "name": "tech_input"}, "inbound_nodes": []}, {"name": "tech_dense_0", "class_name": "Dense", "config": {"name": "tech_dense_0", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["tech_input", 0, 0, {}]]]}, {"name": "lstm_input", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 48, 1], "dtype": "float32", "sparse": false, "name": "lstm_input"}, "inbound_nodes": []}, {"name": "tech_relu_0", "class_name": "Activation", "config": {"name": "tech_relu_0", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["tech_dense_0", 0, 0, {}]]]}, {"name": "lstm_0", "class_name": "LSTM", "config": {"name": "lstm_0", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "inbound_nodes": [[["lstm_input", 0, 0, {}]]]}, {"name": "tech_dense_3", "class_name": "Dense", "config": {"name": "tech_dense_3", "trainable": true, "dtype": "float32", "units": 25, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["tech_relu_0", 0, 0, {}]]]}, {"name": "lstm_1", "class_name": "LSTM", "config": {"name": "lstm_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 25, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "inbound_nodes": [[["lstm_0", 0, 0, {}]]]}, {"name": "tech_relu_3", "class_name": "Activation", "config": {"name": "tech_relu_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["tech_dense_3", 0, 0, {}]]]}, {"name": "concatenate", "class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["lstm_1", 0, 0, {}], ["tech_relu_3", 0, 0, {}]]]}, {"name": "dense_pooling", "class_name": "Dense", "config": {"name": "dense_pooling", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"name": "dense_pooling_1", "class_name": "Dense", "config": {"name": "dense_pooling_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_pooling", 0, 0, {}]]]}, {"name": "dense_out", "class_name": "Dense", "config": {"name": "dense_out", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_pooling_1", 0, 0, {}]]]}], "input_layers": [["lstm_input", 0, 0], ["tech_input", 0, 0]], "output_layers": [["dense_out", 0, 0]]}, "input_spec": [null, null], "activity_regularizer": null, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_14", "layers": [{"name": "tech_input", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 7], "dtype": "float32", "sparse": false, "name": "tech_input"}, "inbound_nodes": []}, {"name": "tech_dense_0", "class_name": "Dense", "config": {"name": "tech_dense_0", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["tech_input", 0, 0, {}]]]}, {"name": "lstm_input", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 48, 1], "dtype": "float32", "sparse": false, "name": "lstm_input"}, "inbound_nodes": []}, {"name": "tech_relu_0", "class_name": "Activation", "config": {"name": "tech_relu_0", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["tech_dense_0", 0, 0, {}]]]}, {"name": "lstm_0", "class_name": "LSTM", "config": {"name": "lstm_0", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "inbound_nodes": [[["lstm_input", 0, 0, {}]]]}, {"name": "tech_dense_3", "class_name": "Dense", "config": {"name": "tech_dense_3", "trainable": true, "dtype": "float32", "units": 25, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["tech_relu_0", 0, 0, {}]]]}, {"name": "lstm_1", "class_name": "LSTM", "config": {"name": "lstm_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 25, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "inbound_nodes": [[["lstm_0", 0, 0, {}]]]}, {"name": "tech_relu_3", "class_name": "Activation", "config": {"name": "tech_relu_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["tech_dense_3", 0, 0, {}]]]}, {"name": "concatenate", "class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["lstm_1", 0, 0, {}], ["tech_relu_3", 0, 0, {}]]]}, {"name": "dense_pooling", "class_name": "Dense", "config": {"name": "dense_pooling", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"name": "dense_pooling_1", "class_name": "Dense", "config": {"name": "dense_pooling_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_pooling", 0, 0, {}]]]}, {"name": "dense_out", "class_name": "Dense", "config": {"name": "dense_out", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_pooling_1", 0, 0, {}]]]}], "input_layers": [["lstm_input", 0, 0], ["tech_input", 0, 0]], "output_layers": [["dense_out", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "tech_input", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 7], "config": {"batch_input_shape": [null, 7], "dtype": "float32", "sparse": false, "name": "tech_input"}, "input_spec": null, "activity_regularizer": null}
?

kernel
bias
_callable_losses
_eager_losses
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "tech_dense_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "tech_dense_0", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7}}}, "activity_regularizer": null}
?
_callable_losses
 _eager_losses
!trainable_variables
"regularization_losses
#	variables
$	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "lstm_input", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 48, 1], "config": {"batch_input_shape": [null, 48, 1], "dtype": "float32", "sparse": false, "name": "lstm_input"}, "input_spec": null, "activity_regularizer": null}
?
%_callable_losses
&_eager_losses
'trainable_variables
(regularization_losses
)	variables
*	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "tech_relu_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "tech_relu_0", "trainable": true, "dtype": "float32", "activation": "relu"}, "input_spec": null, "activity_regularizer": null}
?

+cell
,
state_spec
-_callable_losses
._eager_losses
/trainable_variables
0regularization_losses
1	variables
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?{"class_name": "LSTM", "name": "lstm_0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_0", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 50, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 1], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "activity_regularizer": null}
?

3kernel
4bias
5_callable_losses
6_eager_losses
7trainable_variables
8regularization_losses
9	variables
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "tech_dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "tech_dense_3", "trainable": true, "dtype": "float32", "units": 25, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "activity_regularizer": null}
?

;cell
<
state_spec
=_callable_losses
>_eager_losses
?trainable_variables
@regularization_losses
A	variables
B	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?{"class_name": "LSTM", "name": "lstm_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 25, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 50], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "activity_regularizer": null}
?
C_callable_losses
D_eager_losses
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "tech_relu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "tech_relu_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "input_spec": null, "activity_regularizer": null}
?
I_callable_losses
J_eager_losses
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "input_spec": null, "activity_regularizer": null}
?

Okernel
Pbias
Q_callable_losses
R_eager_losses
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_pooling", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_pooling", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "activity_regularizer": null}
?

Wkernel
Xbias
Y_callable_losses
Z_eager_losses
[trainable_variables
\regularization_losses
]	variables
^	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_pooling_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_pooling_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "activity_regularizer": null}
?

_kernel
`bias
a_callable_losses
b_eager_losses
ctrainable_variables
dregularization_losses
e	variables
f	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_out", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_out", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "activity_regularizer": null}
?
giter

hbeta_1

ibeta_2
	jdecay
klearning_ratem?m?3m?4m?Om?Pm?Wm?Xm?_m?`m?lm?mm?nm?om?pm?qm?v?v?3v?4v?Ov?Pv?Wv?Xv?_v?`v?lv?mv?nv?ov?pv?qv?"
	optimizer
?
0
1
l2
m3
n4
35
46
o7
p8
q9
O10
P11
W12
X13
_14
`15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
l2
m3
n4
35
46
o7
p8
q9
O10
P11
W12
X13
_14
`15"
trackable_list_wrapper
?
trainable_variables
rnon_trainable_variables
regularization_losses
	variables

slayers
tmetrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
unon_trainable_variables
trainable_variables
regularization_losses
	variables

vlayers
wmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%22tech_dense_0_4/kernel
!:22tech_dense_0_4/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
xnon_trainable_variables
trainable_variables
regularization_losses
	variables

ylayers
zmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
{non_trainable_variables
!trainable_variables
"regularization_losses
#	variables

|layers
}metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
~non_trainable_variables
'trainable_variables
(regularization_losses
)	variables

layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

lkernel
mrecurrent_kernel
nbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_cell_9", "trainable": true, "dtype": "float32", "units": 50, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": null, "activity_regularizer": null}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
l0
m1
n2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
l0
m1
n2"
trackable_list_wrapper
?
?non_trainable_variables
/trainable_variables
0regularization_losses
1	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%22tech_dense_3_4/kernel
!:2tech_dense_3_4/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
?non_trainable_variables
7trainable_variables
8regularization_losses
9	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

okernel
precurrent_kernel
qbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_cell_10", "trainable": true, "dtype": "float32", "units": 25, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": null, "activity_regularizer": null}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
o0
p1
q2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
o0
p1
q2"
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
@regularization_losses
A	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Etrainable_variables
Fregularization_losses
G	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Ktrainable_variables
Lregularization_losses
M	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&22dense_pooling_5/kernel
": 2dense_pooling_5/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
?
?non_trainable_variables
Strainable_variables
Tregularization_losses
U	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(
2dense_pooling_1_4/kernel
$:"
2dense_pooling_1_4/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
?
?non_trainable_variables
[trainable_variables
\regularization_losses
]	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
2dense_out_4/kernel
:2dense_out_4/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
?
?non_trainable_variables
ctrainable_variables
dregularization_losses
e	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
": 	?2lstm_0_4/kernel
,:*	2?2lstm_0_4/recurrent_kernel
:?2lstm_0_4/bias
!:2d2lstm_1_4/kernel
+:)d2lstm_1_4/recurrent_kernel
:d2lstm_1_4/bias
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
l0
m1
n2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
l0
m1
n2"
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
?regularization_losses
?	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
+0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
o0
p1
q2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
o0
p1
q2"
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
?regularization_losses
?	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
;0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?
_fn_kwargs
?_updates
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}, "input_spec": null, "activity_regularizer": null}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?non_trainable_variables
?trainable_variables
?regularization_losses
?	variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
,:*22Adam/tech_dense_0_4/kernel/m
&:$22Adam/tech_dense_0_4/bias/m
,:*22Adam/tech_dense_3_4/kernel/m
&:$2Adam/tech_dense_3_4/bias/m
-:+22Adam/dense_pooling_5/kernel/m
':%2Adam/dense_pooling_5/bias/m
/:-
2Adam/dense_pooling_1_4/kernel/m
):'
2Adam/dense_pooling_1_4/bias/m
):'
2Adam/dense_out_4/kernel/m
#:!2Adam/dense_out_4/bias/m
':%	?2Adam/lstm_0_4/kernel/m
1:/	2?2 Adam/lstm_0_4/recurrent_kernel/m
!:?2Adam/lstm_0_4/bias/m
&:$2d2Adam/lstm_1_4/kernel/m
0:.d2 Adam/lstm_1_4/recurrent_kernel/m
 :d2Adam/lstm_1_4/bias/m
,:*22Adam/tech_dense_0_4/kernel/v
&:$22Adam/tech_dense_0_4/bias/v
,:*22Adam/tech_dense_3_4/kernel/v
&:$2Adam/tech_dense_3_4/bias/v
-:+22Adam/dense_pooling_5/kernel/v
':%2Adam/dense_pooling_5/bias/v
/:-
2Adam/dense_pooling_1_4/kernel/v
):'
2Adam/dense_pooling_1_4/bias/v
):'
2Adam/dense_out_4/kernel/v
#:!2Adam/dense_out_4/bias/v
':%	?2Adam/lstm_0_4/kernel/v
1:/	2?2 Adam/lstm_0_4/recurrent_kernel/v
!:?2Adam/lstm_0_4/bias/v
&:$2d2Adam/lstm_1_4/kernel/v
0:.d2 Adam/lstm_1_4/recurrent_kernel/v
 :d2Adam/lstm_1_4/bias/v
?2?
C__inference_model_14_layer_call_and_return_conditional_losses_76937
C__inference_model_14_layer_call_and_return_conditional_losses_76994
C__inference_model_14_layer_call_and_return_conditional_losses_76867
C__inference_model_14_layer_call_and_return_conditional_losses_76901?
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
 __inference__wrapped_model_70967?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *X?U
S?P
(?%

lstm_input?????????0
$?!

tech_input?????????
?2?
(__inference_model_14_layer_call_fn_77014
(__inference_model_14_layer_call_fn_76957?
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_tech_dense_0_layer_call_and_return_conditional_losses_74752?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_tech_dense_0_layer_call_fn_74763?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_lstm_input_layer_call_and_return_conditional_losses_74791?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_lstm_input_layer_call_fn_74800?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_tech_relu_0_layer_call_and_return_conditional_losses_74774?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_tech_relu_0_layer_call_fn_74783?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_lstm_0_layer_call_and_return_conditional_losses_71435
A__inference_lstm_0_layer_call_and_return_conditional_losses_75297
A__inference_lstm_0_layer_call_and_return_conditional_losses_75764
A__inference_lstm_0_layer_call_and_return_conditional_losses_72846?
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_lstm_0_layer_call_fn_72377
&__inference_lstm_0_layer_call_fn_75782
&__inference_lstm_0_layer_call_fn_75773
&__inference_lstm_0_layer_call_fn_72853?
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_tech_dense_3_layer_call_and_return_conditional_losses_74814?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_tech_dense_3_layer_call_fn_74825?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_lstm_1_layer_call_and_return_conditional_losses_76253
A__inference_lstm_1_layer_call_and_return_conditional_losses_73319
A__inference_lstm_1_layer_call_and_return_conditional_losses_74730
A__inference_lstm_1_layer_call_and_return_conditional_losses_76720?
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_lstm_1_layer_call_fn_74737
&__inference_lstm_1_layer_call_fn_76729
&__inference_lstm_1_layer_call_fn_74261
&__inference_lstm_1_layer_call_fn_76738?
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_tech_relu_3_layer_call_and_return_conditional_losses_76748?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_tech_relu_3_layer_call_fn_76757?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_concatenate_layer_call_and_return_conditional_losses_76768?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_concatenate_layer_call_fn_76778?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_pooling_layer_call_and_return_conditional_losses_76793?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dense_pooling_layer_call_fn_76804?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_dense_pooling_1_layer_call_and_return_conditional_losses_76821?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_dense_pooling_1_layer_call_fn_76832?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_out_layer_call_and_return_conditional_losses_76849?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_out_layer_call_fn_76860?
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B=
#__inference_signature_wrapper_77046
lstm_input
tech_input
?2??
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec!
args?
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
&__inference_lstm_1_layer_call_fn_76738Xopq7?4
-?*
$?!
inputs?????????02
p
? "???????????
(__inference_model_14_layer_call_fn_77014?34lmnopqOPWX_`f?c
\?Y
S?P
(?%

lstm_input?????????0
$?!

tech_input?????????
p
? "???????????
D__inference_dense_out_layer_call_and_return_conditional_losses_76849\_`/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? ?
&__inference_lstm_1_layer_call_fn_74737hopqG?D
=?:
4?1
/?,
inputs/0??????????????????2
p
? "???????????
G__inference_tech_dense_3_layer_call_and_return_conditional_losses_74814\34/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????
? ?
&__inference_lstm_0_layer_call_fn_72377ulmnG?D
=?:
4?1
/?,
inputs/0??????????????????
p 
? "%?"??????????????????2?
C__inference_model_14_layer_call_and_return_conditional_losses_76994?34lmnopqOPWX_`b?_
X?U
O?L
&?#
inputs/0?????????0
"?
inputs/1?????????
p
? "%?"
?
0?????????
? |
)__inference_dense_out_layer_call_fn_76860O_`/?,
%?"
 ?
inputs?????????

? "???????????
 __inference__wrapped_model_70967?34lmnopqOPWX_`b?_
X?U
S?P
(?%

lstm_input?????????0
$?!

tech_input?????????
? "5?2
0
	dense_out#? 
	dense_out??????????
&__inference_lstm_1_layer_call_fn_74261hopqG?D
=?:
4?1
/?,
inputs/0??????????????????2
p 
? "???????????
F__inference_tech_relu_3_layer_call_and_return_conditional_losses_76748X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
A__inference_lstm_1_layer_call_and_return_conditional_losses_76720eopq7?4
-?*
$?!
inputs?????????02
p
? "%?"
?
0?????????
? ?
A__inference_lstm_0_layer_call_and_return_conditional_losses_71435?lmnG?D
=?:
4?1
/?,
inputs/0??????????????????
p 
? "2?/
(?%
0??????????????????2
? z
+__inference_tech_relu_0_layer_call_fn_74783K/?,
%?"
 ?
inputs?????????2
? "??????????2?
G__inference_tech_dense_0_layer_call_and_return_conditional_losses_74752\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????2
? ?
/__inference_dense_pooling_1_layer_call_fn_76832OWX/?,
%?"
 ?
inputs?????????
? "??????????
?
C__inference_model_14_layer_call_and_return_conditional_losses_76901?34lmnopqOPWX_`f?c
\?Y
S?P
(?%

lstm_input?????????0
$?!

tech_input?????????
p
? "%?"
?
0?????????
? z
+__inference_tech_relu_3_layer_call_fn_76757K/?,
%?"
 ?
inputs?????????
? "???????????
J__inference_dense_pooling_1_layer_call_and_return_conditional_losses_76821\WX/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? ?
E__inference_lstm_input_layer_call_and_return_conditional_losses_74791n:?7
0?-
+?(
&?#
inputs/0?????????0
? "0?-
&?#
!?
0/0?????????0
? ?
A__inference_lstm_1_layer_call_and_return_conditional_losses_74730uopqG?D
=?:
4?1
/?,
inputs/0??????????????????2
p
? "%?"
?
0?????????
? ?
&__inference_lstm_0_layer_call_fn_75773\lmn7?4
-?*
$?!
inputs?????????0
p 
? "??????????02
,__inference_tech_dense_0_layer_call_fn_74763O/?,
%?"
 ?
inputs?????????
? "??????????2?
A__inference_lstm_0_layer_call_and_return_conditional_losses_75764ilmn7?4
-?*
$?!
inputs?????????0
p
? ")?&
?
0?????????02
? ?
C__inference_model_14_layer_call_and_return_conditional_losses_76867?34lmnopqOPWX_`f?c
\?Y
S?P
(?%

lstm_input?????????0
$?!

tech_input?????????
p 
? "%?"
?
0?????????
? ?
&__inference_lstm_0_layer_call_fn_75782\lmn7?4
-?*
$?!
inputs?????????0
p
? "??????????02?
A__inference_lstm_1_layer_call_and_return_conditional_losses_76253eopq7?4
-?*
$?!
inputs?????????02
p 
? "%?"
?
0?????????
? ?
F__inference_concatenate_layer_call_and_return_conditional_losses_76768?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????2
? ?
C__inference_model_14_layer_call_and_return_conditional_losses_76937?34lmnopqOPWX_`b?_
X?U
O?L
&?#
inputs/0?????????0
"?
inputs/1?????????
p 
? "%?"
?
0?????????
? ?
-__inference_dense_pooling_layer_call_fn_76804OOP/?,
%?"
 ?
inputs?????????2
? "???????????
*__inference_lstm_input_layer_call_fn_74800b:?7
0?-
+?(
&?#
inputs/0?????????0
? "$?!
?
0?????????0?
+__inference_concatenate_layer_call_fn_76778vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "??????????2?
#__inference_signature_wrapper_77046?34lmnopqOPWX_`y?v
? 
o?l
2

tech_input$?!

tech_input?????????
6

lstm_input(?%

lstm_input?????????0"5?2
0
	dense_out#? 
	dense_out??????????
A__inference_lstm_0_layer_call_and_return_conditional_losses_75297ilmn7?4
-?*
$?!
inputs?????????0
p 
? ")?&
?
0?????????02
? ?
&__inference_lstm_0_layer_call_fn_72853ulmnG?D
=?:
4?1
/?,
inputs/0??????????????????
p
? "%?"??????????????????2
,__inference_tech_dense_3_layer_call_fn_74825O34/?,
%?"
 ?
inputs?????????2
? "???????????
H__inference_dense_pooling_layer_call_and_return_conditional_losses_76793\OP/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????
? ?
&__inference_lstm_1_layer_call_fn_76729Xopq7?4
-?*
$?!
inputs?????????02
p 
? "???????????
(__inference_model_14_layer_call_fn_76957?34lmnopqOPWX_`f?c
\?Y
S?P
(?%

lstm_input?????????0
$?!

tech_input?????????
p 
? "???????????
A__inference_lstm_0_layer_call_and_return_conditional_losses_72846?lmnG?D
=?:
4?1
/?,
inputs/0??????????????????
p
? "2?/
(?%
0??????????????????2
? ?
A__inference_lstm_1_layer_call_and_return_conditional_losses_73319uopqG?D
=?:
4?1
/?,
inputs/0??????????????????2
p 
? "%?"
?
0?????????
? ?
F__inference_tech_relu_0_layer_call_and_return_conditional_losses_74774X/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????2
? 