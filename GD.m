function [obj_GD, loss_GD, transmitted_bits]=GD...
    (XX,YY, no_workers, num_feature, noSamples, num_iter, obj0, lambda_logistic)


   
s1=num_feature;
s2=noSamples;
grads=ones(num_feature,no_workers);
alpha = 0.0001;
out_central=zeros(s1,1);



max_iter = num_iter;


 for i = 1:max_iter
     
     transmitted_bits(i) = i*num_feature*32;
    
     for ii =1:no_workers

       
         first = (ii-1)*s2+1;
         last = first+s2-1;
        
        
         grads(:,ii)=-(XX(first:last,1:num_feature)'*(YY(first:last)./(1+exp(YY(first:last).*(XX(first:last,1:num_feature)*out_central)))))+lambda_logistic*out_central;
  
                   
     end
        
    
    out_central=out_central-alpha*sum(grads,2)/no_workers;%inv(sum(hessian,3))*sum(grads,2);



        %final_obj = 0;
        final_obj =lambda_logistic*0.5*norm(out_central)^2;
        for ii =1:no_workers
            first = (ii-1)*s2+1;
            last = first+s2-1;
            %final_obj = final_obj + 0.5*norm(XX(first:last,1:s1)*out_central - YY(first:last))^2;
            final_obj = final_obj+sum(log(1+exp(-YY(first:last).*(XX(first:last,1:s1)*out_central))));
        end
        %i
        obj_GD(i)=final_obj;
        
        final_obj
        abs(final_obj-obj0)
        
        
        
        loss_GD(i)=abs(final_obj-obj0);

            

        
        
    end   
    
end
     
