% Auxillary plot formatting function
function PlotFormatting(plot_title, label_x, label_y, label_z, legendFlag, legLabels, legLoc, fontSize)
    grid on;
    set(gca,'TickLabelInterpreter','latex','FontSize',fontSize)
    title(plot_title,'Interpreter','latex','FontSize',fontSize);
    xlabel(label_x,'Interpreter','latex','FontSize',fontSize);
    ylabel(label_y,'Interpreter','latex','FontSize',fontSize);
    zlabel(label_z,'Interpreter','latex','FontSize',fontSize);
    if legendFlag
        legend(legLabels,'Interpreter','latex','FontSize',fontSize,...
               'location',legLoc);
    end
end