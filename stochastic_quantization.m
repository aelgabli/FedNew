function [quantized]=stochastic_quantization(quantized,current,prev,bitsToSend)


b=bitsToSend;
tau=1/(2^b-1);
%number_of_bits_toSend =32+length(current)*b;% the number of bits to send the value of R.

diff=current - prev;
R=max(abs(diff));
    
% Stochastic Quantization
Q=(diff+R)/(2*tau*R);
p=(Q-floor(Q));
for i=1:length(current)
    temp=rand;
    if(temp <=p(i))
        Q(i)=ceil(Q(i));
    else
        Q(i)=floor(Q(i));
    end    
end
quantized=quantized+2*tau*Q*R-R;

end 