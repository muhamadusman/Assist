close all
clear all
clc

addpath('/home/andek67/Research_projects/nifti_matlab/');

basepath_nii = '/local/data1/andek67/ProgressiveGAN/rawdata/BRATS_2020/MICCAI_BraTS2020_TrainingData/';
%basepath_nii = '/local/data1/andek67/ProgressiveGAN/rawdata/BRATS_2021/';

% Get list of subjects
subjects = dir(basepath_nii);

% Remove all non-subjects
removeIndex = 1;
removeIndices = [];
for subject = 1:length(subjects)
    if not(contains(subjects(subject).name,'BraTS2020')) % BraTS2021    
        removeIndices(removeIndex) = subject;
        removeIndex = removeIndex + 1;
    end
end
subjects(removeIndices) = [];

% Do a random permutation of subjects, to get training data from all sites
permutation = randperm(length(subjects));
subjects = subjects(permutation);

load subjectPermutation_BRATS2020.mat
%save('subjectPermutation_BRATS2021.mat','permutation','subjects')
%load subjectPermutation_BRATS2021.mat

% Need to do all splits in one go, because otherwise randperm will give different divisions for different settings...
% We are saving the exact divisions in subjectPermutation.mat

% Get the last 56 subjects for test
%subjects = subjects(314:end); % brats2020
%subjects = subjects(1196:end); % brats2021

%teststring = '_test';
teststring = '';

for augmentation = 1:1

    if augmentation == 1
        useAugmentation = false;
    else
        useAugmentation = true;
    end
    
    for channels = 1:1
        if channels == 1
            separateSegmentationChannels = false;
        else
            separateSegmentationChannels = true;
        end               

        slicesPerSubject = zeros(313,1);    
        maxT1PerSubject = zeros(313,1);    

        for numberOfSubjects = [313] % 313  1195
                        
            if useAugmentation && not(separateSegmentationChannels)
                dataset = ['brats2021_' num2str(numberOfSubjects) 'subjects_rotationaugmentation_singlesegmentationchannel_min15percentcoverage' teststring];
            elseif not(useAugmentation) && not(separateSegmentationChannels)
                dataset = ['brats2021_' num2str(numberOfSubjects) 'subjects_noaugmentation_singlesegmentationchannel_min15percentcoverage' teststring];
            elseif useAugmentation && separateSegmentationChannels
                dataset = ['brats2021_' num2str(numberOfSubjects) 'subjects_rotationaugmentation_multiplesegmentationchannels_min15percentcoverage' teststring];
            elseif not(useAugmentation) && separateSegmentationChannels
                dataset = ['brats2021_' num2str(numberOfSubjects) 'subjects_noaugmentation_multiplesegmentationchannels_min15percentcoverage' teststring];
            end
            
            dataset
            
            basepath_png = ['/local/data1/andek67/ProgressiveGAN/downloads/' dataset '/'];
            
            for subject = 1:numberOfSubjects

                subject
                
                nii = load_nii([basepath_nii subjects(subject).name '/' subjects(subject).name '_t1.nii']);
                volume_T1 = nii.img; volume_T1 = double(volume_T1);
                nii = load_nii([basepath_nii subjects(subject).name '/' subjects(subject).name '_t2.nii']);
                volume_T2 = nii.img; volume_T2 = double(volume_T2);
                nii = load_nii([basepath_nii subjects(subject).name '/' subjects(subject).name '_t1ce.nii']);
                volume_T1ce = nii.img; volume_T1ce = double(volume_T1ce);
                nii = load_nii([basepath_nii subjects(subject).name '/' subjects(subject).name '_flair.nii']);
                volume_flair = nii.img; volume_flair = double(volume_flair);
                nii = load_nii([basepath_nii subjects(subject).name '/' subjects(subject).name '_seg.nii']);
                volume_seg = nii.img; volume_seg = double(volume_seg);
                
                [sy sx sz] = size(volume_T1);
                
                subject
                
                maxT1PerSubject(subject) = max(volume_T1(:));

                % Normalize intensity to 0 - 255 (8 bit)
                volume_T1 = volume_T1 / max(volume_T1(:)) * 255;
                volume_T2 = volume_T2 / max(volume_T2(:)) * 255;
                volume_T1ce = volume_T1ce / max(volume_T1ce(:)) * 255;
                volume_flair = volume_flair / max(volume_flair(:)) * 255;
                
                % For separate channels we want to keep the original values, to then split the annotations using these exact values
                if not(separateSegmentationChannels)
                    volume_seg = volume_seg / 5 * 255; % Because the label values should be the same for all subjects
                end
                
                includedSlices = 0;
                
                for z = 1:sz
                    slice_T1 = volume_T1(:,:,z);
                    slice_T2 = volume_T2(:,:,z);
                    slice_T1ce = volume_T1ce(:,:,z);
                    slice_flair = volume_flair(:,:,z);
                    slice_seg = volume_seg(:,:,z);
                    
                    % Pad to 256 x 256 (for GAN training)
                    temp = zeros(256,256); temp(9:end-8,9:end-8) = slice_T1; slice_T1 = temp;
                    temp = zeros(256,256); temp(9:end-8,9:end-8) = slice_T2; slice_T2 = temp;
                    temp = zeros(256,256); temp(9:end-8,9:end-8) = slice_T1ce; slice_T1ce = temp;
                    temp = zeros(256,256); temp(9:end-8,9:end-8) = slice_flair; slice_flair = temp;
                    temp = zeros(256,256); temp(9:end-8,9:end-8) = slice_seg; slice_seg = temp;
                    
                    if separateSegmentationChannels
                        % Separate annotations to different channels
                        slice_seg_onehot = zeros(256,256,3);
                        slice_seg_onehot(:,:,1) = 255*ones(256,256) .* (slice_seg == 1);
                        slice_seg_onehot(:,:,2) = 255*ones(256,256) .* (slice_seg == 2);
                        slice_seg_onehot(:,:,3) = 255*ones(256,256) .* (slice_seg == 4);
                        slice_seg_onehot = double(slice_seg_onehot);
                    end
                    
                    highpixels = sum(slice_T1(:) > 50);
                    
                    % Ignore slices with too much black
                    if highpixels/(256*256) > 0.15
                        includedSlices = includedSlices + 1;
                        
                        %subjectChars = 3-1; % Brats 2020
                        subjectChars = 4-1; % Brats 2021

                        %imwrite(uint8(slice_T1),[basepath_png 'T1/' 'Subject_' subjects(subject).name(end-subjectChars:end) '_slice_' num2str(z) '_T1.png'],'png');
                        %imwrite(uint8(slice_T2),[basepath_png 'T2/' 'Subject_' subjects(subject).name(end-subjectChars:end) '_slice_' num2str(z) '_T2.png'],'png');
                        %imwrite(uint8(slice_T1ce),[basepath_png 'T1CE/' 'Subject_' subjects(subject).name(end-subjectChars:end) '_slice_' num2str(z) '_T1ce.png'],'png');
                        %imwrite(uint8(slice_flair),[basepath_png 'FLAIR/' 'Subject_' subjects(subject).name(end-subjectChars:end) '_slice_' num2str(z) '_flair.png'],'png');
                        
                        if not(separateSegmentationChannels)
                            %imwrite(uint8(slice_seg),[basepath_png 'Seg/' 'Subject_' subjects(subject).name(end-subjectChars:end) '_slice_' num2str(z) '_seg.png'],'png');
                        else
                            %imwrite(uint8(slice_seg_onehot(:,:,1)),[basepath_png 'Seg1/' 'Subject_' subjects(subject).name(end-subjectChars:end) '_slice_' num2str(z) '_seg_channel1.png'],'png');
                            %imwrite(uint8(slice_seg_onehot(:,:,2)),[basepath_png 'Seg2/' 'Subject_' subjects(subject).name(end-subjectChars:end) '_slice_' num2str(z) '_seg_channel2.png'],'png');
                            %imwrite(uint8(slice_seg_onehot(:,:,3)),[basepath_png 'Seg3/' 'Subject_' subjects(subject).name(end-subjectChars:end) '_slice_' num2str(z) '_seg_channel3.png'],'png');
                        end
                        
                        if useAugmentation
                            
                            for augmentation = 1:9
                                randomRotation = 150*rand - 75; % Random rotation between -75 and 75, uniform distribution
                                rotated_T1 = imrotate(slice_T1,randomRotation,'bilinear','crop');
                                rotated_T2 = imrotate(slice_T2,randomRotation,'bilinear','crop');
                                rotated_T1ce = imrotate(slice_T1ce,randomRotation,'bilinear','crop');
                                rotated_flair = imrotate(slice_flair,randomRotation,'bilinear','crop');
                                
                                %imwrite(uint8(rotated_T1),[basepath_png 'T1/' 'Subject_' subjects(subject).name(end-subjectChars:end) '_slice_' num2str(z) '_augmentation_' num2str(augmentation) '_T1.png'],'png');
                                %imwrite(uint8(rotated_T2),[basepath_png 'T2/' 'Subject_' subjects(subject).name(end-subjectChars:end) '_slice_' num2str(z) '_augmentation_' num2str(augmentation) '_T2.png'],'png');
                                %imwrite(uint8(rotated_T1ce),[basepath_png 'T1CE/' 'Subject_' subjects(subject).name(end-subjectChars:end) '_slice_' num2str(z) '_augmentation_' num2str(augmentation) '_T1ce.png'],'png');
                                %imwrite(uint8(rotated_flair),[basepath_png 'FLAIR/' 'Subject_' subjects(subject).name(end-subjectChars:end) '_slice_' num2str(z) '_augmentation_' num2str(augmentation) '_flair.png'],'png');
                                
                                if not(separateSegmentationChannels)
                                    rotated_seg = imrotate(slice_seg,randomRotation,'nearest','crop'); % nearest for annotations
                                    %imwrite(uint8(rotated_seg),[basepath_png 'Seg/' 'Subject_' subjects(subject).name(end-subjectChars:end) '_slice_' num2str(z) '_augmentation_' num2str(augmentation) '_seg.png'],'png');
                                else
                                    rotated_seg1 = imrotate(slice_seg_onehot(:,:,1),randomRotation,'nearest','crop'); % nearest for annotations
                                    rotated_seg2 = imrotate(slice_seg_onehot(:,:,2),randomRotation,'nearest','crop'); % nearest for annotations
                                    rotated_seg3 = imrotate(slice_seg_onehot(:,:,3),randomRotation,'nearest','crop'); % nearest for annotations
                                    %imwrite(uint8(rotated_seg1),[basepath_png 'Seg1/' 'Subject_' subjects(subject).name(end-subjectChars:end) '_slice_' num2str(z) '_augmentation_' num2str(augmentation) '_seg_channel1.png'],'png');
                                    %imwrite(uint8(rotated_seg2),[basepath_png 'Seg2/' 'Subject_' subjects(subject).name(end-subjectChars:end) '_slice_' num2str(z) '_augmentation_' num2str(augmentation) '_seg_channel2.png'],'png');
                                    %imwrite(uint8(rotated_seg3),[basepath_png 'Seg3/' 'Subject_' subjects(subject).name(end-subjectChars:end) '_slice_' num2str(z) '_augmentation_' num2str(augmentation) '_seg_channel3.png'],'png');                                    
                                end
                            end
                        end
                    end
                    
                end
                slicesPerSubject(subject) = includedSlices;
                includedSlices
                maxT1PerSubject(subject)
            end
        end
    end
end

slicesPerSubject


%slice_T1 = slice_T1 / (max(slice_T1(:)) + eps) * 255;
%slice_T2 = slice_T2 / (max(slice_T2(:)) + eps) * 255;
%slice_T1ce = slice_T1ce / (max(slice_T1ce(:)) + eps) * 255;
%slice_flair = slice_flair / (max(slice_flair(:)) + eps) * 255;
%slice_seg = slice_seg / 5 * 255; % Because the values should be the same for all subjects

