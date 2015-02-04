function writetofile(y)
    fid = fopen('classes_nn.txt','wt');

    for ii = 1:size(y,1)
        fprintf(fid,'%g\t',y(ii,:));
        fprintf(fid,'\n');
    end
    fclose(fid);
end