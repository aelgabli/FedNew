function [obj_GD, loss_GD,transmitted_bits]=newton_QADMM_Hk...
    (XX,YY, no_workers, num_feature, noSamples, num_iter, obj0, lambda_logistic, bitsToSend, rho, alpha)

  
s1=num_feature;
s2=noSamples;

%rho=500;
%alpha = 0;

grads=ones(num_feature,no_workers);
hessian = zeros(num_feature,num_feature, no_workers);
prev_w=zeros(s1,no_workers);
q_w=zeros(s1,no_workers);
w_central = zeros(num_feature,1);
w=zeros(num_feature,no_workers);
lambda=zeros(num_feature,no_workers);

out_central=zeros(s1,1);


max_iter = num_iter;
R=1;


 for i = 1:max_iter
%     if i==1
%         transmitted_bits(i)=num_feature*bitsToSend+32;
%     else
%         transmitted_bits(i)=transmitted_bits(i-1)+num_feature*bitsToSend+32;
%     end
     
    transmitted_bits(i) = i*R*(num_feature*bitsToSend + 32);


     for ii =1:no_workers

       
         first = (ii-1)*s2+1;
         last = first+s2-1;
        

         %grads(:,ii)=XX(first:last,1:num_feature)'*XX(first:last,1:num_feature)*out_central-XX(first:last,1:num_feature)'*YY(first:last);
         
         grads(:,ii)=-(XX(first:last,1:num_feature)'*(YY(first:last)./(1+exp(YY(first:last).*(XX(first:last,1:num_feature)*out_central)))))+lambda_logistic*out_central;
        


         
         %hessian(:,:,ii)= XX(first:last,1:num_feature)'*XX(first:last,1:num_feature);
         
         
         
         %if (i==1)
             
             temp = (lambda_logistic + alpha)*eye(num_feature,num_feature);

             for jj=first:last
                 temp=temp+YY(jj)^2*XX(jj,:)'*XX(jj,:)*(exp(YY(jj)*XX(jj,:)*out_central)/(1+exp(YY(jj)*XX(jj,:)*out_central))^2);

             end

             hessian(:,:,ii)=temp;
         %end
         

         
     end
     
     
     for r=1:R
        for ii=1:no_workers

            

                %temp2= (hessian(:,:,ii)+(rho)*eye(num_feature,num_feature))\(grads(:,ii)-lambda(:,ii)+rho*w_central);
                
                temp2= pinv(hessian(:,:,ii)+(rho)*eye(num_feature,num_feature))*(grads(:,ii)-lambda(:,ii)+rho*w_central);
                


            w(:,ii) =temp2;
                                              
            [q_w(:,ii)]=stochastic_quantization(q_w(:,ii),w(:,ii),prev_w(:,ii),bitsToSend);
            


            w(:,ii) = q_w(:,ii);
            prev_w(:,ii) = q_w(:,ii);


        end
        
        
            w_central = (rho * sum(w,2)+sum(lambda,2))/(no_workers*rho);
            
        
        for ii=1:no_workers
            lambda(:,ii)=lambda(:,ii)+rho*(w(:,ii)-w_central);
        end
     end
        


     


    
    out_central=out_central-w_central;
    



        %final_obj = 0;
        final_obj =lambda_logistic*0.5*norm(out_central)^2;
        for ii =1:no_workers
            first = (ii-1)*s2+1;
            last = first+s2-1;
            %final_obj = final_obj + 0.5*norm(XX(first:last,1:s1)*out_central - YY(first:last))^2;
            final_obj = final_obj+sum(log(1+exp(-YY(first:last).*(XX(first:last,1:s1)*out_central))));
        end

        obj_GD(i)=final_obj;
        final_obj
        loss_GD(i)=abs(final_obj-obj0);
        abs(final_obj-obj0)
        



    end   
    
end
     
